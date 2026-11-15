//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/typed/initial_state.h>

namespace ryujin
{
  namespace EulerInitialStates
  {
    /**
     * The low Mach number Riemann problem of @cite DegondTang2011: a
     * base state \f$(\rho_0, m_0)\f$ on \f$[0,1]\f$ perturbed along the
     * first coordinate direction,
     * \f{align*}
     *   (\rho, m) = \begin{cases}
     *   (\rho_0, m_0 - \varepsilon^2/2) & x < 0.2 \text{ or } x \ge 0.8,
     *   \\
     *   (\rho_0 + \varepsilon^2, m_0) & 0.2 \le x < 0.3, \\
     *   (\rho_0, m_0 + \varepsilon^2/2) & 0.3 \le x < 0.7, \\
     *   (\rho_0 - \varepsilon^2, m_0) & 0.7 \le x < 0.8.
     *   \end{cases}
     * \f}
     *
     * The pressure (if the system has an energy equation) is the one of
     * the base state.
     *
     * @note The @p t argument is ignored. This class always returns the
     * initial configuration.
     *
     * @ingroup EulerEquations
     */
    template <typename HyperbolicSystem, int dim, typename Number>
    class DegondRiemann : public InitialState<HyperbolicSystem, dim, Number>
    {
    public:
      using View = typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename View::state_type;

      DegondRiemann(const HyperbolicSystem &hyperbolic_system,
                    const std::string &subsection)
          : InitialState<HyperbolicSystem, dim, Number>(
                "degond riemann problem", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        density_ = 1.;
        this->add_parameter("density", density_, "Density of the base state");

        momentum_ = 1.;
        this->add_parameter(
            "momentum", momentum_, "Momentum of the base state");

        pressure_ = 1.;
        if constexpr (View::have_energy_equation)
          this->add_parameter(
              "pressure", pressure_, "Pressure of the base state");

        mach_number_ = 1.;
        if constexpr (!View::have_mach_number) {
          this->add_parameter("mach number",
                              mach_number_,
                              "Mach number epsilon of the perturbation");
        }
      }

      state_type compute(const dealii::Point<dim> &point, Number /*t*/) final
      {
        const auto view = hyperbolic_system_.template view<dim, Number>();

        if constexpr (View::have_mach_number) {
          mach_number_ = view.mach_number();
        }

        const Number x = point[0];
        const Number eps2 = mach_number_ * mach_number_;

        Number rho = density_;
        Number m = momentum_ - Number(0.5) * eps2;
        if (x >= Number(0.2) && x < Number(0.3)) {
          rho += eps2;
          m = momentum_;
        } else if (x >= Number(0.3) && x < Number(0.7)) {
          m = momentum_ + Number(0.5) * eps2;
        } else if (x >= Number(0.7) && x < Number(0.8)) {
          rho -= eps2;
          m = momentum_;
        }

        state_type primitive_state;
        primitive_state[0] = rho;
        primitive_state[1] = m / rho;
        if constexpr (View::have_energy_equation)
          primitive_state[dim + 1] = pressure_;

        return view.from_primitive_state(primitive_state);
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;

      Number density_;
      Number momentum_;
      Number pressure_;
      Number mach_number_;
    };
  } // namespace EulerInitialStates
} // namespace ryujin
