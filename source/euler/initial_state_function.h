//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2024 by the ryujin authors
//

#pragma once

#include <initial_state_library.h>

#include <deal.II/base/function_parser.h>

namespace ryujin
{
  namespace EulerInitialStates
  {
    /**
     * Returns an initial state defined by a set of user specified function.
     *
     * @ingroup EulerEquations
     */
    template <typename Description, int dim, typename Number>
    class Function : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using View =
          typename Description::template HyperbolicSystemView<dim, Number>;
      using state_type = typename View::state_type;

      Function(const HyperbolicSystem &hyperbolic_system,
               const std::string subsection)
          : InitialState<Description, dim, Number>("function", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {

        density_expression_ = "1.4";
        this->add_parameter("density expression",
                            density_expression_,
                            "A function expression describing the density");

        velocity_x_expression_ = "3.0";
        this->add_parameter(
            "velocity x expression",
            velocity_x_expression_,
            "A function expression describing the x-component of the velocity");

        if constexpr (dim > 1) {
          velocity_y_expression_ = "0.0";
          this->add_parameter("velocity y expression",
                              velocity_y_expression_,
                              "A function expression describing the "
                              "y-component of the velocity");
        }

        if constexpr (dim > 2) {
          velocity_z_expression_ = "0.0";
          this->add_parameter("velocity z expression",
                              velocity_z_expression_,
                              "A function expression describing the "
                              "z-component of the velocity");
        }

        pressure_expression_ = "1.0";
        this->add_parameter("pressure expression",
                            pressure_expression_,
                            "A function expression describing the pressure");

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
          density_function_ = std::make_unique<FP>(density_expression_);
          velocity_x_function_ = std::make_unique<FP>(velocity_x_expression_);
          if constexpr (dim > 1)
            velocity_y_function_ = std::make_unique<FP>(velocity_y_expression_);
          if constexpr (dim > 2)
            velocity_z_function_ = std::make_unique<FP>(velocity_z_expression_);
          pressure_function_ = std::make_unique<FP>(pressure_expression_);
        };

        set_up_muparser();
        this->parse_parameters_call_back.connect(set_up_muparser);
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        const auto view = hyperbolic_system_.template view<dim, Number>();
        state_type full_primitive_state;

        density_function_->set_time(t);
        full_primitive_state[0] = density_function_->value(point);

        velocity_x_function_->set_time(t);
        full_primitive_state[1] = velocity_x_function_->value(point);

        if constexpr (dim > 1) {
          velocity_y_function_->set_time(t);
          full_primitive_state[2] = velocity_y_function_->value(point);
        }
        if constexpr (dim > 2) {
          velocity_z_function_->set_time(t);
          full_primitive_state[3] = velocity_z_function_->value(point);
        }

        pressure_function_->set_time(t);
        full_primitive_state[1 + dim] = pressure_function_->value(point);

        return view.from_primitive_state(full_primitive_state);
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;

      std::string density_expression_;
      std::string velocity_x_expression_;
      std::string velocity_y_expression_;
      std::string velocity_z_expression_;
      std::string pressure_expression_;

      std::unique_ptr<dealii::FunctionParser<dim>> density_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> velocity_x_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> velocity_y_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> velocity_z_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> pressure_function_;
    };
  } // namespace EulerInitialStates
} // namespace ryujin