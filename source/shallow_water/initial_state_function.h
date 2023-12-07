//
// SPDX-License-Identifier: MIT or BSD-3-Clause
// [LANL Copyright Statement]
// Copyright (C) 2020 - 2023 by the ryujin authors
// Copyright (C) 2023 - 2023 by Triad National Security, LLC
//

#pragma once

#include <initial_state_library.h>

#include <deal.II/base/function_parser.h>

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
    class Function : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using HyperbolicSystemView =
          typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      Function(const HyperbolicSystem &hyperbolic_system,
               const std::string subsection)
          : InitialState<Description, dim, Number>("function", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {

        depth_expression_ = "1.4";
        this->add_parameter("water depth expression",
                            depth_expression_,
                            "A function expression describing the water depth");

        velocity_expression_ = "3.0";
        this->add_parameter("velocity expression",
                            velocity_expression_,
                            "A function expression describing the velocity");

        /*
         * Set up the muparser object with the final flux description from
         * the parameter file:
         */
        const auto set_up_muparser = [this] {
          using FP = dealii::FunctionParser<dim>;
          /*
           * This variant of the constructor initializes the function
           * parser with support for a time-dependent description involving
           * a variable »t«:
           */
          depth_function_ = std::make_unique<FP>(depth_expression_);
          velocity_function_ = std::make_unique<FP>(velocity_expression_);
        };

        set_up_muparser();
        this->parse_parameters_call_back.connect(set_up_muparser);
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        dealii::Tensor<1, 2, Number> primitive;

        depth_function_->set_time(t);
        primitive[0] = depth_function_->value(point);
        velocity_function_->set_time(t);
        primitive[1] = velocity_function_->value(point);

        return hyperbolic_system_.from_initial_state(primitive);
      }

    private:
      const HyperbolicSystemView hyperbolic_system_;

      bool use_primitive_state_functions_;

      std::string depth_expression_;
      std::string velocity_expression_;
      std::string bathymetry_expression_;

      std::unique_ptr<dealii::FunctionParser<dim>> depth_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> velocity_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> bathymetry_function_;
    };
  } // namespace ShallowWaterInitialStates
} // namespace ryujin
