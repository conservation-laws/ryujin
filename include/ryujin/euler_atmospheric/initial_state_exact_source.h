//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// [LANL Copyright Statement]
// Copyright (C) 2025 - 2026 by the ryujin authors
// Copyright (C) 2025 - 2026 by Triad National Security, LLC
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/typed/initial_state.h>

#include <cmath>

namespace ryujin
{
  namespace EulerAtmosphericInitialStates
  {
    /**
     * An exact solution with gravitational source: constant density
     * \f$\rho_0\f$, constant vertical velocity \f$v_0\f$, and a potential
     * temperature profile in hydrostatic balance that is advected with
     * \f$v_0\f$:
     * \f{align}
     *   \theta(\boldsymbol x,t)^\gamma \;=\;
     *   -\frac{g}{K\rho_0^{\gamma-1}}\,(x_d - v_0 t),
     * \f}
     * where \f$K\f$ is the constant returned by eos_constant(). The
     * integration constant is set to zero, so the computational domain has
     * to satisfy \f$x_d - v_0 t \le 0\f$.
     *
     * @note This class returns the analytic solution as a function of time
     * @p t and position @p x.
     *
     * @ingroup EulerAtmosphericEquations
     */
    template <typename HyperbolicSystem, int dim, typename Number>
    class ExactSource : public InitialState<HyperbolicSystem, dim, Number>
    {
    public:
      using View = typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename View::state_type;

      ExactSource(const HyperbolicSystem &hyperbolic_system,
                  const std::string subsection)
          : InitialState<HyperbolicSystem, dim, Number>("exact source",
                                                        subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        density_ref_ = 1.;
        this->add_parameter(
            "reference density", density_ref_, "The constant density");

        velocity_ = 1.;
        this->add_parameter(
            "velocity", velocity_, "The constant vertical velocity");
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        const auto view = hyperbolic_system_.template view<dim, Number>();

        const Number x = point[dim - 1] - velocity_ * t;
        const Number theta_pow_gamma =
            -view.gravity() /
            (view.eos_constant() *
             std::pow(density_ref_, view.gamma_minus_one())) *
            x;

        state_type primitive_state;
        primitive_state[0] = density_ref_;
        primitive_state[dim] = velocity_;
        primitive_state[dim + 1] =
            std::pow(theta_pow_gamma, view.gamma_inverse());

        return view.from_primitive_state(primitive_state);
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

      Number density_ref_;
      Number velocity_;
    };
  } // namespace EulerAtmosphericInitialStates
} // namespace ryujin
