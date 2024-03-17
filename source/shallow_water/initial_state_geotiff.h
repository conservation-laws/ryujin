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
     * Returns an initial state defined by a set of user specified functions
     * based on the primitive variables.
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
            "filename", filename_, "GeoTIFF image file to load");

        height_expression_ = "1.4";
        this->add_parameter(
            "water height expression",
            height_expression_,
            "A function expression describing the water height");

        velocity_expression_ = "0.0";
        this->add_parameter("velocity expression",
                            velocity_expression_,
                            "A function expression describing the velocity");

        const auto set_up = [this] {
#ifdef WITH_GDAL
          /* Initial GDAL and reset all data: */
          GDALAllRegister();
          driver_name = "";
          driver_projection = "";
          affine_transformation = {0, 0, 0, 0, 0, 0};
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
        primitive[0] = std::max(Number(0.), height_function_->value(point) - z);

        velocity_function_->set_time(t);
        primitive[1] = velocity_function_->value(point);

        const auto view = hyperbolic_system_.template view<dim, Number>();
        return view.from_initial_state(primitive);
      }

      typename InitialState<Description, dim, Number>::precomputed_state_type
      initial_precomputations(const dealii::Point<dim> &point) final
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
         * Construct the affine transformation from information in the geotiff
         * image, or from the parameter file:
         */

        const auto tiff_with_geotransform =
            dataset->GetGeoTransform(affine_transformation.data()) == CE_None;

//         printf("Origin = (%.6f,%.6f)\n",
//                affine_transformation[0],
//                affine_transformation[3]);
//         printf("Pixel Size = (%.6f,%.6f)\n",
//                affine_transformation[1],
//                affine_transformation[5]);
//         printf("Pixel Rota = (%.6f,%.6f)\n",
//                affine_transformation[2],
//                affine_transformation[4]);

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

        GDALClose(dataset_handle);

#ifdef DEBUG_OUTPUT
        std::cout << "GDAL: driver name    = " << driver_name;
        std::cout << "\nGDAL: projection     = " << driver_projection;
        std::cout << "\nGDAL: transformation =";
        for (const auto &it : affine_transformation)
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


      DEAL_II_ALWAYS_INLINE inline Number
      compute_bathymetry(const dealii::Point<dim> & /*point*/) const
      {
        geotiff.ensure_initialized([&]() { read_in_raster(); return true; });

        return 0.;
      }


      /* Runtime parameters: */

      std::string filename_;

      std::string height_expression_;
      std::string velocity_expression_;

      /* GDAL data structures: */

      //
      // We use a Lazy<t> wrapper for lazy initialization with efficient
      // Schmidt's double checking. We simply ignore the bool type here.
      //
      Lazy<bool> geotiff;

      mutable std::string driver_name;
      mutable std::string driver_projection;
      mutable std::array<double, 6> affine_transformation;
      mutable std::array<int, 2> raster_offset;
      mutable std::array<int, 2> raster_size;
      mutable std::vector<float> raster;

      /* Fields for muparser support for water height and velocity: */

      std::unique_ptr<dealii::FunctionParser<dim>> height_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> velocity_function_;
    };
  } // namespace ShallowWaterInitialStates
} // namespace ryujin
