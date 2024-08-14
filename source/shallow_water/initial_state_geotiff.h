//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2024 by the ryujin authors
//

#pragma once

#include <initial_state_library.h>
#include <lazy.h>

#include <deal.II/base/function_parser.h>


#ifdef WITH_GDAL
#include <cpl_conv.h>
#include <gdal.h>
#include <gdal_priv.h>
#endif

namespace ryujin
{
  namespace ShallowWaterInitialStates
  {
    /**
     * Returns an initial state by reading a bathymetry from a geotiff
     * file. For this we link against GDAL, see https://gdal.org/index.html
     * for details on GDAL and what image formats it supports.
     *
     * @ingroup ShallowWaterEquations
     */
    template <typename Description, int dim, typename Number>
    class GeoTIFF : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using View =
          typename Description::template HyperbolicSystemView<dim, Number>;
      using state_type = typename View::state_type;


      GeoTIFF(const HyperbolicSystem &hyperbolic_system,
              const std::string subsection)
          : InitialState<Description, dim, Number>("geotiff", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        filename_ = "ryujin.tif";
        this->add_parameter(
            "filename", filename_, "GeoTIFF: image file to load");

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

        height_expression_ = "1.4";
        this->add_parameter(
            "water height expression",
            height_expression_,
            "A function expression describing the initial total water height");

        velocity_expression_ = "0.0";
        this->add_parameter(
            "velocity expression",
            velocity_expression_,
            "A function expression describing the initial velocity");

        const auto set_up = [this] {
#ifdef WITH_GDAL
          /* Initial GDAL and reset all data: */
          GDALAllRegister();
          driver_name = "";
          driver_projection = "";
          affine_transformation = {0, 0, 0, 0, 0, 0};
          inverse_affine_transformation = {0, 0, 0, 0, 0, 0};
          raster_offset = {0, 0};
          raster_size = {0, 0};
          raster.clear();
#endif

          using FP = dealii::FunctionParser<dim>;
          /*
           * This variant of the constructor initializes the function
           * parser with support for a time-dependent description involving
           * a variable »t«:
           */
          height_function_ = std::make_unique<FP>(height_expression_);
          velocity_function_ = std::make_unique<FP>(velocity_expression_);
        };

        set_up();
        this->parse_parameters_call_back.connect(set_up);
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        const auto z = compute_bathymetry(point);

        dealii::Tensor<1, 2, Number> primitive;

        height_function_->set_time(t);
        primitive[0] = std::max(0., height_function_->value(point) - z);

        velocity_function_->set_time(t);
        primitive[1] = velocity_function_->value(point);

        const auto view = hyperbolic_system_.template view<dim, Number>();
        return view.from_initial_state(primitive);
      }

      auto initial_precomputations(const dealii::Point<dim> &point) ->
          typename InitialState<Description, dim, Number>::
              initial_precomputed_type final
      {
        /* Compute bathymetry: */
        return {compute_bathymetry(point)};
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;


      void read_in_raster() const
      {
#ifdef WITH_GDAL
        auto dataset_handle = GDALOpen(filename_.c_str(), GA_ReadOnly);
        AssertThrow(dataset_handle,
                    dealii::ExcMessage("GDAL error: file not found"));

        auto dataset = GDALDataset::FromHandle(dataset_handle);
        Assert(dataset, dealii::ExcInternalError());

        const auto driver = dataset->GetDriver();

        driver_name = driver->GetMetadataItem(GDAL_DMD_LONGNAME);
        if (dataset->GetProjectionRef() != nullptr)
          driver_projection = dataset->GetProjectionRef();

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

        raster_offset = {0, 0};
        raster_size = {dataset->GetRasterXSize(), dataset->GetRasterYSize()};

        raster.resize(raster_size[0] * raster_size[1]);
        const auto error_code = raster_band->RasterIO(
            GF_Read,
            raster_offset[0], /* x-offset of image region */
            raster_offset[1], /* y-offset of image region */
            raster_size[0],   /* x-size of image region */
            raster_size[1],   /* y-size of image region */
            raster.data(),
            raster_size[0], /* x-size of target buffer */
            raster_size[1], /* y-size of target buffer */
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
              dataset->GetGeoTransform(affine_transformation.data()) == CE_None;
          AssertThrow(success,
                      dealii::ExcMessage("GDAL driver error: no geo transform "
                                         "present in geotiff file"));
        } else {
          affine_transformation = transformation_;
          /* Flip sign for j index (y-coordinate): */
          affine_transformation[2] *= -1.;
          affine_transformation[5] *= -1.;
        }

        /*
         * Ensure that (i=0, j=raster_size[1]-1) corresponds to the user
         * supplied (transformation_[0], transformation_[3]).
         */
        if (transformation_use_geotiff_ == false ||
            transformation_use_geotiff_origin_ == false) {
          const auto j_max = raster_size[1] - 1;
          affine_transformation[0] =
              transformation_[0] - j_max * affine_transformation[2];
          affine_transformation[3] =
              transformation_[3] - j_max * affine_transformation[5];
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
        inverse_affine_transformation[0] = affine_transformation[0];
        inverse_affine_transformation[3] = affine_transformation[3];

        const auto determinant =
            affine_transformation[1] * affine_transformation[5] -
            affine_transformation[2] * affine_transformation[4];
        const auto inv = 1. / determinant;
        inverse_affine_transformation[1] = inv * affine_transformation[5];
        inverse_affine_transformation[2] = inv * (-affine_transformation[2]);
        inverse_affine_transformation[4] = inv * (-affine_transformation[4]);
        inverse_affine_transformation[5] = inv * affine_transformation[1];

        GDALClose(dataset_handle);

#ifdef DEBUG_OUTPUT
        std::cout << std::setprecision(16);
        std::cout << "GDAL: driver name    = " << driver_name;
        std::cout << "\nGDAL: projection     = " << driver_projection;
        std::cout << "\nGDAL: transformation =";
        for (const auto &it : affine_transformation)
          std::cout << " " << it;
        std::cout << "\nGDAL: inverse trafo =";
        for (const auto &it : inverse_affine_transformation)
          std::cout << " " << it;
        std::cout << "\nGDAL: raster offset  =";
        for (const auto &it : raster_offset)
          std::cout << " " << it;
        std::cout << "\nGDAL: raster size    =";
        for (const auto &it : raster_size)
          std::cout << " " << it;
        std::cout << std::endl;
#endif

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
        const auto &at = affine_transformation;
        const double x = at[0] + at[1] * i + at[2] * j;
        const double y = at[3] + at[4] * i + at[5] * j;
        return {x, y};
      }


      DEAL_II_ALWAYS_INLINE inline std::array<double, 2>
      apply_inverse_transformation(const double x, const double y) const
      {
        const auto &iat = inverse_affine_transformation;
        const double i = iat[1] * (x - iat[0]) + iat[2] * (y - iat[3]);
        const double j = iat[4] * (x - iat[0]) + iat[5] * (y - iat[3]);
        return {i, j};
      }


      DEAL_II_ALWAYS_INLINE inline Number
      compute_bathymetry(const dealii::Point<dim> &point) const
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

        /*
         * Use a simple bilinear interpolation:
         */

        const auto i_left = static_cast<unsigned int>(std::floor(di));
        const auto i_right = static_cast<unsigned int>(std::ceil(di));
        const auto j_left = static_cast<unsigned int>(std::floor(dj));
        const auto j_right = static_cast<unsigned int>(std::ceil(dj));

        const bool in_bounds =
            i_left <= i_right &&
            i_right < static_cast<unsigned int>(raster_size[0]) &&
            j_left <= j_right &&
            j_right < static_cast<unsigned int>(raster_size[1]);

        AssertThrow(
            in_bounds,
            dealii::ExcMessage("Raster error: The requested point is outside "
                               "the image boundary of the geotiff file"));

        const double i_ratio = std::fmod(di, 1.);
        const double j_ratio = std::fmod(dj, 1.);

        const auto v_iljl = raster[i_left + j_left * raster_size[0]];
        const auto v_irjl = raster[i_right + j_left * raster_size[0]];

        const auto v_iljr = raster[i_left + j_right * raster_size[0]];
        const auto v_irjr = raster[i_right + j_right * raster_size[0]];

        const auto v_jl = v_iljl * (1. - i_ratio) + v_irjl * i_ratio;
        const auto v_jr = v_iljr * (1. - i_ratio) + v_irjr * i_ratio;

        return v_jl * (1. - j_ratio) + v_jr * j_ratio;
      }


      /* Runtime parameters: */

      std::string filename_;

      std::array<double, 6> transformation_;
      bool transformation_use_geotiff_;
      bool transformation_use_geotiff_origin_;

      std::string height_expression_;
      std::string velocity_expression_;

      /* GDAL data structures: */

      //
      // We use a Lazy<t> wrapper for lazy initialization with efficient
      // Schmidt's double checking. We simply ignore the bool type here.
      //
      Lazy<bool> geotiff_guard_;
      mutable std::string driver_name;
      mutable std::string driver_projection;
      mutable std::array<double, 6> affine_transformation;
      mutable std::array<double, 6> inverse_affine_transformation;
      mutable std::array<int, 2> raster_offset;
      mutable std::array<int, 2> raster_size;
      mutable std::vector<float> raster;

      /* Fields for muparser support for water height and velocity: */

      std::unique_ptr<dealii::FunctionParser<dim>> height_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> velocity_function_;
    };
  } // namespace ShallowWaterInitialStates
} // namespace ryujin
