//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <geotiff_reader.h>
#include <initial_state_library.h>
#include <lazy.h>

#include <deal.II/base/function_parser.h>

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
      using View = typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename View::state_type;


      GeoTIFF(const HyperbolicSystem &hyperbolic_system,
              const std::string subsection)
          : InitialState<Description, dim, Number>("geotiff", subsection)
          , hyperbolic_system_(hyperbolic_system)
          , geotiff_reader_(subsection + "/geotiff")
      {
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
        const auto z = geotiff_reader_.compute_height(point);

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
        return {static_cast<Number>(geotiff_reader_.compute_height(point))};
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;
      mutable GeoTIFFReader geotiff_reader_;

      /* Runtime parameters: */

      std::string height_expression_;
      std::string velocity_expression_;

      /* Fields for muparser support for water height and velocity: */

      std::unique_ptr<dealii::FunctionParser<dim>> height_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> velocity_function_;
    };
  } // namespace ShallowWaterInitialStates
} // namespace ryujin
