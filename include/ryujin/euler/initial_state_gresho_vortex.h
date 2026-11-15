//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/typed/initial_state.h>

#include <numbers>

namespace ryujin
{
  namespace EulerInitialStates
  {
    /**
     * The Gresho vortex: a stationary solution of the incompressible
     * Euler equations with density \f$\rho = 1\f$, the angular velocity
     * \f[
     *   u_\varphi(r) = \begin{cases}
     *     5r & r < 0.2, \\ 2-5r & 0.2\le r<0.4, \\ 0 & r \ge 0.4,
     *   \end{cases}
     * \f]
     * and the pressure \f$p = p_0 + p_1(r)\f$ with
     * \f[
     *   p_1(r) = \begin{cases}
     *     \tfrac{25}{2}r^2 & r < 0.2, \\
     *     \tfrac{25}{2}r^2 + 4\big(1 - 5r + \log(5r)\big) & 0.2\le r<0.4,
     *     \\ -2 + 4\log 2 & r \ge 0.4,
     *   \end{cases}
     * \f]
     * where the background pressure \f$p_0 = 1/(\gamma M^2) - 1/2\f$
     * sets the maximal Mach number \f$M\f$. The vortex lies in the
     * \f$x_1x_2\f$ plane and is advected with the constant velocity
     * \f$U\f$ in \f$x_1\f$ direction, i.e., it is centered at \f$(Ut,
     * 0)\f$ with the velocity \f$U\boldsymbol e_1\f$ added.
     *
     * @note This class returns the analytic solution as a function of time
     * @p t and position @p x.
     *
     * @ingroup EulerEquations
     */
    template <typename HyperbolicSystem, int dim, typename Number>
    class GreshoVortex : public InitialState<HyperbolicSystem, dim, Number>
    {
    public:
      using View = typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename View::state_type;

      GreshoVortex(const HyperbolicSystem &hyperbolic_system,
                   const std::string &subsection)
          : InitialState<HyperbolicSystem, dim, Number>("gresho vortex",
                                                        subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        gamma_ = 1.4;
        if constexpr (!View::have_gamma) {
          this->add_parameter("gamma", gamma_, "The ratio of specific heats");
        }

        mach_number_ = 0.001;
        this->add_parameter(
            "mach number", mach_number_, "Maximal Mach number of the vortex");

        advection_velocity_ = 0.;
        this->add_parameter("advection velocity",
                            advection_velocity_,
                            "Constant advection velocity in x direction");
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        AssertThrow(dim > 1, dealii::ExcNotImplemented());

        const auto view = hyperbolic_system_.template view<dim, Number>();

        if constexpr (View::have_gamma) {
          gamma_ = view.gamma();
        }

        state_type primitive_state;
        primitive_state[0] = Number(1.);

        if constexpr (dim > 1) {
          const Number x = point[0] - advection_velocity_ * t;
          const Number y = point[1];
          const Number r = std::sqrt(x * x + y * y);

          Number u_phi = Number(0.);
          Number p_1 = Number(-2.) + Number(4.) * std::numbers::ln2_v<Number>;
          if (r < Number(0.2)) {
            u_phi = Number(5.) * r;
            p_1 = Number(12.5) * r * r;
          } else if (r < Number(0.4)) {
            u_phi = Number(2.) - Number(5.) * r;
            p_1 = Number(12.5) * r * r +
                  Number(4.) *
                      (Number(1.) - Number(5.) * r + std::log(Number(5.) * r));
          }

          primitive_state[1] = advection_velocity_;
          if (r > Number(0.)) {
            primitive_state[1] -= u_phi * y / r;
            primitive_state[2] = u_phi * x / r;
          }

          const Number p_0 =
              Number(1.) / (gamma_ * mach_number_ * mach_number_) - Number(0.5);
          if constexpr (View::have_energy_equation)
            primitive_state[dim + 1] = p_0 + p_1;
        }

        return view.from_primitive_state(primitive_state);
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;

      Number gamma_;
      Number mach_number_;
      Number advection_velocity_;
    };
  } // namespace EulerInitialStates
} // namespace ryujin
