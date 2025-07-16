//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"
#include "lazy.h"
#include "patterns_conversion.h"

#include <deal.II/base/parameter_acceptor.h>
#include <iomanip>

#ifdef WITH_GDAL
#include <cpl_conv.h>
#include <gdal.h>
#include <gdal_priv.h>
#endif

namespace ryujin
{
  enum class HeightNormalization {
    /** do not normalize the height: **/
    none,
    /** set minimum to zero: **/
    minimum,
    /** set average to zero: **/
    average,
    /** set maximum to zero: **/
    maximum
  };
} // namespace ryujin

#ifndef DOXYGEN
DECLARE_ENUM(ryujin::HeightNormalization,
             LIST({ryujin::HeightNormalization::none, "none"},
                  {ryujin::HeightNormalization::minimum, "minimum"},
                  {ryujin::HeightNormalization::average, "average"},
                  {ryujin::HeightNormalization::maximum, "maximum"}));
#endif

namespace ryujin
{
  /**
   * A simple GeoTIFF reader based on the GDAL library. See
   * https://gdal.org/index.html for details on GDAL and what image
   * formats it supports.
   *
   * @ingroup ShallowWaterEquations
   */
  template <int dim>
  class GeoTIFFReader : dealii::ParameterAcceptor
  {
  public:
    GeoTIFFReader(const std::string subsection)
        : ParameterAcceptor(subsection)
    {
      filename_ = "ryujin.tif";
      this->add_parameter("filename", filename_, "GeoTIFF: image file to load");

      transformation_ = {0., 0.01, 0., 0., 0., 0.01};
      this->add_parameter(
          "transformation",
          transformation_,
          "Array \"t[]\" describing an affine transformation between image "
          "space (indices i and j from bottom left) and real coordinates (x "
          "and y): x = t[0] + t[1] * i + t[2] * j, and y = t[3] + t[4] * i + "
          "t[5] * j. (This transformation sets the origin of the image space "
          "into the bottom left corner with index i to the right and index j "
          "up)");

      transformation_allow_out_of_bounds_queries_ = false;
      this->add_parameter(
          "transformation allow out of bounds queries",
          transformation_allow_out_of_bounds_queries_,
          "GeoTIFF: allow out-of-bounds queries. When set to true, the reader "
          "returns constant extended values for coordinates that are outside "
          "of the image range.");

      transformation_use_geotiff_ = true;
      this->add_parameter("transformation use geotiff",
                          transformation_use_geotiff_,
                          "GeoTIFF: read in transformation from GeoTIFF for "
                          "constructing the affine transformation. If set to "
                          "false the manually specified transformation "
                          "parameters will be used instead.");

      transformation_use_geotiff_origin_ = false;
      this->add_parameter(
          "transformation use geotiff origin",
          transformation_use_geotiff_origin_,
          "GeoTIFF: read in affine shift (i.e., position of "
          "lower left corner) from GeoTIFF for constructing "
          "the affine transformation. If set to false the origin specified "
          "in the transformation parameter will be used instead.");

      height_normalization_ = HeightNormalization::minimum;
      this->add_parameter("height normalization",
                          height_normalization_,
                          "GeoTIFF: choose base point for height normalization "
                          "that is set to 0.: none, minimum, average, maximum");

      height_scaling_ = 1.0;
      this->add_parameter("height scaling",
                          height_scaling_,
                          "GeoTIFF: choose base point for height normalization "
                          "that is set to 0.: none, minimum, average, maximum");

      const auto set_up = [&] {
#ifdef WITH_GDAL
        /* Initial GDAL and reset all data: */
        GDALAllRegister();
        driver_name_ = "";
        driver_projection_ = "";
        affine_transformation_ = {0, 0, 0, 0, 0, 0};
        inverse_affine_transformation_ = {0, 0, 0, 0, 0, 0};
        raster_offset_ = {0, 0};
        raster_size_ = {0, 0};
        raster_.clear();
#endif
      };

      set_up();
      this->parse_parameters_call_back.connect(set_up);
    }


    DEAL_II_ALWAYS_INLINE inline double
    compute_height(const dealii::Point<dim> &point) const
    {
      geotiff_guard_.ensure_initialized([&]() {
        read_in_raster();
        return true;
      });

      const double x = point[0];
      double y = 0;
      if constexpr (dim >= 2)
        y = point[1];
      const auto &[di, dj] = apply_inverse_transformation(x, y);

      /* Check that we are in bounds: */

      const bool in_bounds =
          di > -1.0 && di < static_cast<double>(raster_size_[0]) + 1.0 &&
          dj > -1.0 && dj < static_cast<double>(raster_size_[1]) + 1.0;

#ifdef DEBUG_OUTPUT
      if (!in_bounds) {
        std::cout << std::setprecision(16);
        std::cout << "Queried point out of bounds." << std::endl;
        std::cout << "Point: " << point << std::endl;
        std::cout << "Transformed coordinates: (" << di << "," << dj << ")"
                  << std::endl;
      }
#endif

      AssertThrow(
          transformation_allow_out_of_bounds_queries_ || in_bounds,
          dealii::ExcMessage("Raster error: The requested point is outside "
                             "the image boundary of the geotiff file"));

      /*
       * Use a simple bilinear interpolation and ensure we never go below
       * the minimum or above the maximum index.
       */

      const auto i_left = std::min(
          std::max(static_cast<int>(std::floor(di)), 0), raster_size_[0]);
      const auto i_right = std::min(
          std::max(static_cast<int>(std::ceil(di)), 0), raster_size_[0]);
      const auto j_left = std::min(
          std::max(static_cast<int>(std::floor(dj)), 0), raster_size_[1]);
      const auto j_right = std::min(
          std::max(static_cast<int>(std::ceil(dj)), 0), raster_size_[1]);

#ifdef DEBUG_OUTPUT
      if (!in_bounds) {
        std::cout << "index bounding box: (" << i_left << "," << j_left
                  << ") and (" << i_right << "," << j_right << ")" << std::endl;
      }
#endif

      const double i_ratio = std::fmod(di, 1.);
      const double j_ratio = std::fmod(dj, 1.);

      const auto v_iljl = raster_[i_left + j_left * raster_size_[0]];
      const auto v_irjl = raster_[i_right + j_left * raster_size_[0]];

      const auto v_iljr = raster_[i_left + j_right * raster_size_[0]];
      const auto v_irjr = raster_[i_right + j_right * raster_size_[0]];

      const auto v_jl = v_iljl * (1. - i_ratio) + v_irjl * i_ratio;
      const auto v_jr = v_iljr * (1. - i_ratio) + v_irjr * i_ratio;

      return height_scaling_ * (v_jl * (1. - j_ratio) + v_jr * j_ratio);
    }

    /*
     * Return the affine transformation information that is stored in the
     * GeoTIFF image.
     */
    ACCESSOR_READ_ONLY(affine_transformation);

    ACCESSOR_READ_ONLY(raster_size);
    ACCESSOR_READ_ONLY(raster_offset);
    ACCESSOR_READ_ONLY(height_scaling);

  private:
    void read_in_raster() const
    {
#ifdef WITH_GDAL
      auto dataset_handle = GDALOpen(filename_.c_str(), GA_ReadOnly);
      AssertThrow(dataset_handle,
                  dealii::ExcMessage("GDAL error: file not found"));

      auto dataset = GDALDataset::FromHandle(dataset_handle);
      Assert(dataset, dealii::ExcInternalError());

      const auto driver = dataset->GetDriver();

      driver_name_ = driver->GetMetadataItem(GDAL_DMD_LONGNAME);
      if (dataset->GetProjectionRef() != nullptr)
        driver_projection_ = dataset->GetProjectionRef();

      /* For now we support one raster in the dataset: */

      AssertThrow(
          dataset->GetRasterCount() == 1,
          dealii::ExcMessage(
              "GDAL driver error: currently we only support one raster"));

      const auto raster_band = dataset->GetRasterBand(1);

      AssertThrow(dataset->GetRasterXSize() == raster_band->GetXSize() &&
                      dataset->GetRasterYSize() == raster_band->GetYSize(),
                  dealii::ExcMessage(
                      "GDAL driver error: the raster band has a different "
                      "dimension than the (global) raster dimension of the "
                      "geotiff image. This is not supported."));

      /*
       * FIXME: For now, we simply read in the entire geotiff file on
       * each rank. In order to save memory for very large files it would
       * be possible to create a bounding box for the all active cells of
       * the triangulation and then only read in a small region for which
       * we actually need data.
       */

      raster_offset_ = {0, 0};
      raster_size_ = {dataset->GetRasterXSize(), dataset->GetRasterYSize()};

      raster_.resize(raster_size_[0] * raster_size_[1]);
      const auto error_code = raster_band->RasterIO(
          GF_Read,
          raster_offset_[0], /* x-offset of image region */
          raster_offset_[1], /* y-offset of image region */
          raster_size_[0],   /* x-size of image region */
          raster_size_[1],   /* y-size of image region */
          raster_.data(),
          raster_size_[0], /* x-size of target buffer */
          raster_size_[1], /* y-size of target buffer */
          GDT_Float32,
          0,
          0);

      AssertThrow(error_code == 0,
                  dealii::ExcMessage(
                      "GDAL driver error: error reading in geotiff file"));

      /*
       * Read in the affine transformation from the geotiff image.
       *
       * Note that this transformation differs from the one we use in the
       * parameter file: GDAL uses typical image orientation: the origin
       * of the dataset is in the "top left" corner (instead of bottom
       * left) and the first (column) index goes to the right and the
       * second (row) index goes down.
       */

      if (transformation_use_geotiff_) {
        const auto success =
            dataset->GetGeoTransform(affine_transformation_.data()) == CE_None;
        AssertThrow(success,
                    dealii::ExcMessage("GDAL driver error: no geo transform "
                                       "present in geotiff file"));
      } else {
        affine_transformation_ = transformation_;
        /* Flip sign for j index (y-coordinate): */
        affine_transformation_[2] *= -1.;
        affine_transformation_[5] *= -1.;
      }

      /*
       * Ensure that (i=0, j=raster_size[1]-1) corresponds to the user
       * supplied (transformation_[0], transformation_[3]).
       */
      if (transformation_use_geotiff_ == false ||
          transformation_use_geotiff_origin_ == false) {
        const auto j_max = raster_size_[1] - 1;
        affine_transformation_[0] =
            transformation_[0] - j_max * affine_transformation_[2];
        affine_transformation_[3] =
            transformation_[3] - j_max * affine_transformation_[5];
      }

      /*
       * Compute inverse transformation of
       *
       *    x = t[0] + t[1] * i + t[2] * j, y = t[3] + t[4] * i + t[5] * j.
       *
       * namely:
       *
       *     i =  it[1] * (x - it[0]) + it[2] * (y - it[3])
       *     j =  it[4] * (x - it[0]) + it[5] * (y - it[3])
       */
      inverse_affine_transformation_[0] = affine_transformation_[0];
      inverse_affine_transformation_[3] = affine_transformation_[3];

      const auto determinant =
          affine_transformation_[1] * affine_transformation_[5] -
          affine_transformation_[2] * affine_transformation_[4];
      const auto inv = 1. / determinant;
      inverse_affine_transformation_[1] = inv * affine_transformation_[5];
      inverse_affine_transformation_[2] = inv * (-affine_transformation_[2]);
      inverse_affine_transformation_[4] = inv * (-affine_transformation_[4]);
      inverse_affine_transformation_[5] = inv * affine_transformation_[1];

      GDALClose(dataset_handle);

#ifdef DEBUG_OUTPUT
      std::cout << std::setprecision(16);
      std::cout << "GDAL: driver name    = " << driver_name_;
      std::cout << "\nGDAL: projection     = " << driver_projection_;
      std::cout << "\nGDAL: transformation =";
      for (const auto &it : affine_transformation_)
        std::cout << " " << it;
      std::cout << "\nGDAL: inverse trafo =";
      for (const auto &it : inverse_affine_transformation_)
        std::cout << " " << it;
      std::cout << "\nGDAL: raster offset  =";
      for (const auto &it : raster_offset_)
        std::cout << " " << it;
      std::cout << "\nGDAL: raster size    =";
      for (const auto &it : raster_size_)
        std::cout << " " << it;
      std::cout << std::endl;
#endif

      if (height_normalization_ != HeightNormalization::none) {
        float shift = 0.;

        if (height_normalization_ == HeightNormalization::minimum)
          shift = *std::min_element(std::begin(raster_), std::end(raster_));
        else if (height_normalization_ == HeightNormalization::maximum)
          shift = *std::max_element(std::begin(raster_), std::end(raster_));
        else {
          Assert(height_normalization_ == HeightNormalization::average,
                 dealii::ExcInternalError());
          const auto sum = std::reduce(std::begin(raster_), std::end(raster_));
          shift = sum / raster_.size();
        }

        std::for_each(std::begin(raster_),
                      std::end(raster_),
                      [&](auto &element) { element -= shift; });
      }

#else
      static constexpr auto message =
          "ryujin has to be configured with GDAL support in order to read in "
          "GeoTIFF images";
      AssertThrow(false, dealii::ExcMessage(message));
      __builtin_trap();
#endif
    }


    DEAL_II_ALWAYS_INLINE inline std::array<double, 2>
    apply_transformation(const double i, const double j) const
    {
      const auto &at = affine_transformation_;
      const double x = at[0] + at[1] * i + at[2] * j;
      const double y = at[3] + at[4] * i + at[5] * j;
      return {x, y};
    }


    DEAL_II_ALWAYS_INLINE inline std::array<double, 2>
    apply_inverse_transformation(const double x, const double y) const
    {
      const auto &iat = inverse_affine_transformation_;
      const double i = iat[1] * (x - iat[0]) + iat[2] * (y - iat[3]);
      const double j = iat[4] * (x - iat[0]) + iat[5] * (y - iat[3]);
      return {i, j};
    }

    /* Runtime parameters: */

    std::string filename_;

    std::array<double, 6> transformation_;
    bool transformation_allow_out_of_bounds_queries_;
    bool transformation_use_geotiff_;
    bool transformation_use_geotiff_origin_;
    HeightNormalization height_normalization_;
    double height_scaling_;

    /* GDAL data structures: */

    //
    // We use a Lazy<t> wrapper for lazy initialization with efficient
    // Schmidt's double checking. We simply ignore the bool type here.
    //
    Lazy<bool> geotiff_guard_;
    mutable std::string driver_name_;
    mutable std::string driver_projection_;
    mutable std::array<double, 6> affine_transformation_;
    mutable std::array<double, 6> inverse_affine_transformation_;
    mutable std::array<int, 2> raster_offset_;
    mutable std::array<int, 2> raster_size_;
    mutable std::vector<float> raster_;
  };
} // namespace ryujin
