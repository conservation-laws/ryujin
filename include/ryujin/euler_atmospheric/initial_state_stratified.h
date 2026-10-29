//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// [LANL Copyright Statement]
// Copyright (C) 2020 - 2026 by the ryujin authors
// Copyright (C) 2026 by Triad National Security, LLC
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/typed/initial_state.h>

#include <deal.II/base/point.h>

namespace ryujin
{
  namespace EulerAtmosphericInitialStates
  {
    /**
     * A stratified atmosphere initial state with user-prescribed potential
     * temperature and velocity profiles. The density is computed to enforce
     * hydrostatic balance given the theta profile. All fields are specified
     * as muparser expressions that may depend on spatial coordinates and
     * time.
     *
     * @ingroup EulerAtmosphericEquations
     */
    template <typename HyperbolicSystem, int dim, typename Number>
    class Stratified : public InitialState<HyperbolicSystem, dim, Number>
    {
    public:
      using View = typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename View::state_type;

      Stratified(const HyperbolicSystem &hyperbolic_system,
                 const std::string &subsection)
          : InitialState<HyperbolicSystem, dim, Number>("stratified",
                                                        subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        velocity_x_expression_ = "0.0";
        this->add_parameter("velocity x expression",
                            velocity_x_expression_,
                            "A function expression for the x-velocity");

        if constexpr (dim > 1) {
          velocity_y_expression_ = "0.0";
          this->add_parameter("velocity y expression",
                              velocity_y_expression_,
                              "A function expression for the y-velocity");
        }

        if constexpr (dim > 2) {
          velocity_z_expression_ = "0.0";
          this->add_parameter("velocity z expression",
                              velocity_z_expression_,
                              "A function expression for the z-velocity");
        }

        theta_expression_ = "300.0";
        this->add_parameter(
            "theta expression",
            theta_expression_,
            "A function expression for the potential temperature");

        const auto set_up_muparser = [this] {
          using FP = dealii::FunctionParser<dim>;
          velocity_x_function_ = std::make_unique<FP>(velocity_x_expression_);
          if constexpr (dim > 1)
            velocity_y_function_ = std::make_unique<FP>(velocity_y_expression_);
          if constexpr (dim > 2)
            velocity_z_function_ = std::make_unique<FP>(velocity_z_expression_);
          theta_function_ = std::make_unique<FP>(theta_expression_);
        };

        set_up_muparser();
        this->parse_parameters_call_back.connect(set_up_muparser);
      }

      auto compute(const dealii::Point<dim> &point, Number t)
          -> state_type final
      {
        const auto view = hyperbolic_system_.template view<dim, Number>();

        const Number p_0 = view.reference_pressure();
        const Number gamma = view.gamma();
        const Number gamma_minus_one = gamma - 1.;
        const Number g = view.gravity();
        const Number c_v = view.c_v();

        theta_function_->set_time(t);
        const Number theta = theta_function_->value(point);

        const Number T = theta - (g * point[dim - 1]) / (gamma * c_v);
        const Number pressure =
            p_0 * std::pow(T / theta, gamma / gamma_minus_one);
        const Number rho = pressure / (gamma_minus_one * c_v * T);

        state_type primitive;
        primitive[0] = rho;

        velocity_x_function_->set_time(t);
        primitive[1] = velocity_x_function_->value(point);

        if constexpr (dim > 1) {
          velocity_y_function_->set_time(t);
          primitive[2] = velocity_y_function_->value(point);
        }

        if constexpr (dim > 2) {
          velocity_z_function_->set_time(t);
          primitive[3] = velocity_z_function_->value(point);
        }

        primitive[1 + dim] = theta;

        return view.from_initial_state(primitive);
      }

      auto initial_precomputations(const dealii::Point<dim> &point) ->
          typename InitialState<HyperbolicSystem, dim, Number>::
              initial_precomputed_type final
      {
        return hyperbolic_system_.initial_precomputations(
            point, compute(point, Number(0.)));
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;

      std::string velocity_x_expression_;
      std::string velocity_y_expression_;
      std::string velocity_z_expression_;

      std::unique_ptr<dealii::FunctionParser<dim>> velocity_x_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> velocity_y_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> velocity_z_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> theta_function_;
      std::string theta_expression_;
    };
  } // namespace EulerAtmosphericInitialStates
} // namespace ryujin
