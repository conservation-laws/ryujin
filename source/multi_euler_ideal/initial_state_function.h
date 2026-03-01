//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2024 by the ryujin authors
// Copyright (C) 2025 by Triad National Security, LLC
//

#pragma once

#include "hyperbolic_system.h"

#include <initial_state_library.h>

#include <deal.II/base/function_parser.h>

#include <array>

namespace ryujin
{
  namespace MultiSpeciesEulerInitialStates
  {
    using namespace MultiSpeciesEuler;

    /**
     * Returns an initial state defined by a set of user specified functions.
     * Supports n_species mass fractions, where the last species mass fraction
     * is computed as Y_{n-1} = 1 - sum(Y_k) for k=0,...,n-2.
     *
     * @ingroup MultiSpeciesEulerEquations
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
        /* Initialize mass fraction expressions for species 0 to n-2 */
        for (unsigned int k = 0; k < n_species - 1; ++k) {
          Y_expressions_[k] =
              std::to_string(Number(1.) / Number(n_species)); /* default */
          this->add_parameter(
              "Y_" + std::to_string(k) + " expression",
              Y_expressions_[k],
              "A function expression describing the mass fraction for species " +
                  std::to_string(k));
        }

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
          for (unsigned int k = 0; k < n_species - 1; ++k)
            Y_functions_[k] = std::make_unique<FP>(Y_expressions_[k]);

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
        const Number rho = density_function_->value(point);

        /* Compute partial densities for species 0 to n-2 */
        Number Y_sum = Number(0.);
        for (unsigned int k = 0; k < n_species - 1; ++k) {
          Y_functions_[k]->set_time(t);
          const Number Y_k = Y_functions_[k]->value(point);
          full_primitive_state[k] = Y_k * rho;
          Y_sum += Y_k;
        }
        /* Last species gets the remaining mass fraction */
        full_primitive_state[n_species - 1] = (Number(1.) - Y_sum) * rho;

        velocity_x_function_->set_time(t);
        full_primitive_state[n_species] = velocity_x_function_->value(point);

        if constexpr (dim > 1) {
          velocity_y_function_->set_time(t);
          full_primitive_state[n_species + 1] =
              velocity_y_function_->value(point);
        }
        if constexpr (dim > 2) {
          velocity_z_function_->set_time(t);
          full_primitive_state[n_species + 2] =
              velocity_z_function_->value(point);
        }

        pressure_function_->set_time(t);
        full_primitive_state[n_species + dim] = pressure_function_->value(point);

        return view.from_primitive_state(full_primitive_state);
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;

      std::array<std::string, n_species - 1> Y_expressions_;
      std::string density_expression_;
      std::string velocity_x_expression_;
      std::string velocity_y_expression_;
      std::string velocity_z_expression_;
      std::string pressure_expression_;

      std::array<std::unique_ptr<dealii::FunctionParser<dim>>, n_species - 1>
          Y_functions_;
      std::unique_ptr<dealii::FunctionParser<dim>> density_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> velocity_x_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> velocity_y_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> velocity_z_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> pressure_function_;
    };
  } // namespace MultiSpeciesEulerInitialStates
} // namespace ryujin
