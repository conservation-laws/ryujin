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
        use_primitive_state_functions_ = true;
        this->add_parameter(
            "use primitive state functions",
            use_primitive_state_functions_,
            "If set to true then the initial state will be constructed with "
            "the density, velocity, and pressure function descriptions. If "
            "false, then density, momentum and (total) energy is used.");

        density_expression_ = "1.4";
        this->add_parameter("density expression",
                            density_expression_,
                            "A function expression describing the density");

        use_anisotropic_velocity_functions_ = false;
        this->add_parameter(
            "use anisotropic velocity functions",
            use_anisotropic_velocity_functions_,
            "If set set to true then the initial state will be constructed "
            "anisotropic velocity functions, i.e., specified for each "
            "coordinate direction separately.");

        // Only valid to have anisotropic velocity functions when
        // use_primitive_state_functions_ == true
        AssertThrow((use_anisotropic_velocity_functions_ &&
                     use_primitive_state_functions_) ||
                        !use_anisotropic_velocity_functions_,
                    dealii::ExcMessage(
                        "It is only valid to specify velocity components "
                        "when use_primitive_state_functions_ is true"));

        velocity_expression_ = "3.0";
        this->add_parameter("velocity expression",
                            velocity_expression_,
                            "A function expression describing the velocity");

        velocity_x_expression_ = "3.0";
        this->add_parameter(
            "velocity x expression",
            velocity_x_expression_,
            "A function expression describing the x-component of the velocity");

        velocity_y_expression_ = "0.0";
        this->add_parameter(
            "velocity y expression",
            velocity_y_expression_,
            "A function expression describing the y-component of the velocity");

        velocity_z_expression_ = "0.0";
        this->add_parameter(
            "velocity z expression",
            velocity_z_expression_,
            "A function expression describing the z-component of the velocity");

        pressure_expression_ = "1.0";
        this->add_parameter("pressure expression",
                            pressure_expression_,
                            "A function expression describing the pressure");

        momentum_expression_ = "4.2";
        this->add_parameter("momentum expression",
                            momentum_expression_,
                            "A function expression describing the momentum");

        energy_expression_ = "8.8";
        this->add_parameter("energy expression",
                            energy_expression_,
                            "A function expression describing the energy");

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
          velocity_function_ = std::make_unique<FP>(velocity_expression_);
          velocity_x_function_ = std::make_unique<FP>(velocity_x_expression_);
          velocity_y_function_ = std::make_unique<FP>(velocity_y_expression_);
          velocity_z_function_ = std::make_unique<FP>(velocity_z_expression_);
          pressure_function_ = std::make_unique<FP>(pressure_expression_);
          momentum_function_ = std::make_unique<FP>(momentum_expression_);
          energy_function_ = std::make_unique<FP>(energy_expression_);
        };

        set_up_muparser();
        this->parse_parameters_call_back.connect(set_up_muparser);
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        const auto view = hyperbolic_system_.template view<dim, Number>();

        if (use_primitive_state_functions_) {
          if (use_anisotropic_velocity_functions_) {
            state_type full_primitive_state;
            density_function_->set_time(t);
            full_primitive_state[0] = density_function_->value(point);
            velocity_x_function_->set_time(t);
            full_primitive_state[1] = velocity_x_function_->value(point);
            if (dim > 1) {
              velocity_y_function_->set_time(t);
              full_primitive_state[2] = velocity_y_function_->value(point);
            }
            if (dim > 2) {
              velocity_z_function_->set_time(t);
              full_primitive_state[3] = velocity_z_function_->value(point);
            }
            pressure_function_->set_time(t);
            full_primitive_state[1 + dim] = pressure_function_->value(point);

            return view.from_primitive_state(full_primitive_state);
          } else {
            dealii::Tensor<1, 3, Number> primitive;
            density_function_->set_time(t);
            primitive[0] = density_function_->value(point);
            velocity_function_->set_time(t);
            primitive[1] = velocity_function_->value(point);
            pressure_function_->set_time(t);
            primitive[2] = pressure_function_->value(point);

            return view.from_initial_state(primitive);
          }

        } else {
          dealii::Tensor<1, 3, Number> state;

          density_function_->set_time(t);
          state[0] = density_function_->value(point);
          momentum_function_->set_time(t);
          state[1] = momentum_function_->value(point);
          energy_function_->set_time(t);
          state[2] = energy_function_->value(point);

          return view.expand_state(state);
        }
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;

      bool use_primitive_state_functions_;
      bool use_anisotropic_velocity_functions_;

      std::string density_expression_;
      std::string velocity_expression_;
      std::string velocity_x_expression_;
      std::string velocity_y_expression_;
      std::string velocity_z_expression_;
      std::string pressure_expression_;
      std::string momentum_expression_;
      std::string energy_expression_;

      std::unique_ptr<dealii::FunctionParser<dim>> density_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> velocity_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> velocity_x_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> velocity_y_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> velocity_z_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> pressure_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> momentum_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> energy_function_;
    };
  } // namespace EulerInitialStates
} // namespace ryujin