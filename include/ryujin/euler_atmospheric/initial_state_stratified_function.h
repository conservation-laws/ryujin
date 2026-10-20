//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// [LANL Copyright Statement]
// Copyright (C) 2026 by the ryujin authors
// Copyright (C) 2025 - 2026 by Triad National Security, LLC
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
     * A modification of the "contrast" and "function" initial state. Now, we
     * have an initial state formed by a contrast of a given "left" and "right"
     * primitive state where the "left" state is "inside" the radius, R,
     * and the "right" state is outside. In this case, we want a stratified
     * atmosphere, so the initial state on the outer radius is prescribed by a
     * stratification function.
     *
     * @note The @p t argument is ignored. This class always returns the
     * initial configuration.
     *
     * @todo Transform a nonzero initial velocity into a radial momentum
     * scaling with 1 / r^(dim -1).
     *
     * @ingroup EulerAtmosphericEquations
     */
    template <typename HyperbolicSystem, int dim, typename Number>
    class StratifiedFunction
        : public InitialState<HyperbolicSystem, dim, Number>
    {
    public:
      using View = typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename View::state_type;
      using state_type_1d =
          typename HyperbolicSystem::template View<1, Number>::state_type;

      StratifiedFunction(const HyperbolicSystem &hyperbolic_system,
                         const std::string &subsection)
          : InitialState<HyperbolicSystem, dim, Number>("stratified function",
                                                        subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        density_expression_ = "1.4";
        this->add_parameter(
            "density expression",
            density_expression_,
            "A function expression describing the inner density");

        velocity_x_expression_ = "3.0";
        this->add_parameter("velocity x expression",
                            velocity_x_expression_,
                            "A function expression describing the x-component "
                            "of the inner velocity");

        if constexpr (dim > 1) {
          velocity_y_expression_ = "0.0";
          this->add_parameter("velocity y expression",
                              velocity_y_expression_,
                              "A function expression describing the "
                              "y-component of the inner velocity");
        }

        if constexpr (dim > 2) {
          velocity_z_expression_ = "0.0";
          this->add_parameter("velocity z expression",
                              velocity_z_expression_,
                              "A function expression describing the "
                              "z-component of the inner velocity");
        }

        theta_expression_ = "1.0";
        this->add_parameter(
            "theta expression",
            theta_expression_,
            "A function expression describing the inner potential temperature");

        /* Set up all muparser objects */
        const auto set_up_muparser = [this] {
          using FP = dealii::FunctionParser<dim>;
          density_function_ = std::make_unique<FP>(density_expression_);
          velocity_x_function_ = std::make_unique<FP>(velocity_x_expression_);
          if constexpr (dim > 1)
            velocity_y_function_ = std::make_unique<FP>(velocity_y_expression_);
          if constexpr (dim > 2)
            velocity_z_function_ = std::make_unique<FP>(velocity_z_expression_);
          theta_function_ = std::make_unique<FP>(theta_expression_);
        };

        set_up_muparser();
        this->parse_parameters_call_back.connect(set_up_muparser);

        radius_ = 0.5;
        this->add_parameter("radius", radius_, "Radius of radial area");

        center_x_ = 20.;
        this->add_parameter("blast width",
                            center_x_,
                            "distance from left side for center of blast");

        center_y_ = 20.;
        this->add_parameter("blast height",
                            center_y_,
                            "Height above ground for center of blast");

        factor_x_ = 1.;
        this->add_parameter("x factor",
                            factor_x_,
                            "Horizontal scaling factor of the inner region");

        factor_y_ = 1.;
        this->add_parameter("y factor",
                            factor_y_,
                            "Vertical scaling factor of the inner region");
      }

      auto compute(const dealii::Point<dim> &point, Number t)
          -> state_type final
      {
        const auto view = hyperbolic_system_.template view<dim, Number>();

        state_type full_primitive_inner;

        /* Set up the outer state such that we obtain a stratified atmosphere:
         */
        const Number p_0 = view.reference_pressure();
        const Number theta = 300.;
        const Number gamma = view.gamma();
        const Number gamma_minus_one = gamma - 1.;
        const Number g = view.gravity();
        const Number c_v = view.c_v();
        const Number T = theta - (g * (point[dim - 1])) / (gamma * c_v);
        const Number pressure =
            p_0 * std::pow(T / theta, gamma / (gamma_minus_one));
        const Number rho = pressure / (gamma_minus_one * c_v * T);

        primitive_outer_[0] = rho;
        primitive_outer_[1] = 0.0;
        primitive_outer_[2] = theta;

        state_outer_ = view.from_initial_state(primitive_outer_);

        /* Set up the inner state from the function expressions: */
        density_function_->set_time(t);
        full_primitive_inner[0] = density_function_->value(point);

        velocity_x_function_->set_time(t);
        full_primitive_inner[1] = velocity_x_function_->value(point);

        if constexpr (dim > 1) {
          velocity_y_function_->set_time(t);
          full_primitive_inner[2] = velocity_y_function_->value(point);
        }
        if constexpr (dim > 2) {
          velocity_z_function_->set_time(t);
          full_primitive_inner[3] = velocity_z_function_->value(point);
        }

        theta_function_->set_time(t);
        full_primitive_inner[1 + dim] = theta_function_->value(point);

        /*
         * Override with the correct inner density such that we have a
         * hydrostatic balance:
         */
        const Number theta_inner = full_primitive_inner[1 + dim];
        const Number T_inner =
            theta_inner - (g * (point[dim - 1])) / (gamma * c_v);
        const Number pressure_inner =
            p_0 * std::pow(T_inner / theta_inner, gamma / (gamma_minus_one));
        const Number density_inner =
            pressure_inner / (gamma_minus_one * c_v * T_inner);

        full_primitive_inner[0] = density_inner;
        state_inner_ = view.from_initial_state(full_primitive_inner);

        /* Set up the center of the inner state: */
        dealii::Point<dim> center;
        center[0] = (point[0] - center_x_) / factor_x_;
        center[1] = (point[1] - center_y_) / factor_y_;

        return (center.norm() > radius_ ? state_outer_ : state_inner_);
      }

      auto initial_precomputations(const dealii::Point<dim> &point) ->
          typename InitialState<HyperbolicSystem, dim, Number>::
              initial_precomputed_type final
      {
        /* Return the vertical coordinate: */
        return {Number(point[dim - 1])};
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;

      state_type_1d primitive_outer_;
      double radius_;
      double center_x_;
      double center_y_;
      double factor_x_;
      double factor_y_;

      state_type state_inner_;
      state_type state_outer_;

      std::string density_expression_;
      std::string velocity_x_expression_;
      std::string velocity_y_expression_;
      std::string velocity_z_expression_;
      std::string theta_expression_;

      std::unique_ptr<dealii::FunctionParser<dim>> density_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> velocity_x_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> velocity_y_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> velocity_z_function_;
      std::unique_ptr<dealii::FunctionParser<dim>> theta_function_;
    };
  } // namespace EulerAtmosphericInitialStates
} // namespace ryujin
