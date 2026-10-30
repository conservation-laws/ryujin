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
     * A modification of the "contrast" initial state. Now, we have an
     * initial state formed by a contrast of a given "left" and "right"
     * primitive state where the "left" state is "inside" the radius, R,
     * and the "right" state is outside. In this case, we want a stratified
     * atmosphere, so the initial state on the outer radius is prescribed by a
     * function.
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
    class StratifiedBlast : public InitialState<HyperbolicSystem, dim, Number>
    {
    public:
      using View = typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename View::state_type;
      using state_type_1d =
          typename HyperbolicSystem::template View<1, Number>::state_type;

      StratifiedBlast(const HyperbolicSystem &hyperbolic_system,
                      const std::string &subsection)
          : InitialState<HyperbolicSystem, dim, Number>("stratified blast",
                                                        subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        primitive_inner_[0] = 1.4;
        primitive_inner_[1] = 0.0;
        primitive_inner_[2] = 1.;
        this->add_parameter(
            "primitive state inner",
            primitive_inner_,
            "Initial 1d primitive state (rho, u, theta) on the inner disk");

        primitive_outer_[0] = 1.4;
        primitive_outer_[1] = 0.0;
        primitive_outer_[2] = 1.;
        this->add_parameter(
            "primitive state outer",
            primitive_outer_,
            "Initial 1d primitive state (rho, u, theta) on the outer annulus");

        radius_ = 0.5;
        this->add_parameter("radius", radius_, "Radius of radial area");

        height_ = 20.;
        this->add_parameter(
            "blast height", height_, "Height above ground for center of blast");

        gravity_initial_ = 0.00981;
        this->add_parameter(
            "gravity",
            gravity_initial_,
            "Gravitational constant used to build the hydrostatic initial "
            "profile. This is independent of the equation gravity parameter "
            "so that the initial stratification can be prescribed even when "
            "the PDE source term uses a different value (including zero).");
      }

      auto compute(const dealii::Point<dim> &point, Number /*t*/)
          -> state_type final
      {
        const auto view = hyperbolic_system_.template view<dim, Number>();

        /* Set up the outer state such that we obtain a stratified atmosphere:
         */
        const Number p_0 = view.reference_pressure();
        const Number theta = primitive_outer_[2];
        const Number gamma = view.gamma();
        const Number gamma_minus_one = gamma - 1.;
        const Number g = gravity_initial_;
        const Number c_v = view.c_v();
        const Number T = theta - (g * (point[dim - 1])) / (gamma * c_v);
        const Number pressure =
            p_0 * std::pow(T / theta, gamma / (gamma_minus_one));
        const Number rho = pressure / (gamma_minus_one * c_v * T);

        primitive_outer_[0] = rho;
        primitive_outer_[1] = 0;

        /*
         * Inner state from the parameter file. Replace the density with rho
         * from above so that the density matches on the inner and outer
         * radius:
         */
        state_inner_ = view.from_initial_state(primitive_inner_);
        state_inner_[0] = rho;
        state_inner_[dim + 1] = rho * primitive_inner_[2];

        state_outer_ = view.from_initial_state(primitive_outer_);

        dealii::Point<dim> center;
        center[0] = point[0];
        center[1] = point[1] - height_;

        return (center.norm() > radius_ ? state_outer_ : state_inner_);
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

      state_type_1d primitive_inner_;
      state_type_1d primitive_outer_;
      double radius_;
      double height_;
      double gravity_initial_;

      state_type state_inner_;
      state_type state_outer_;
    };
  } // namespace EulerAtmosphericInitialStates
} // namespace ryujin
