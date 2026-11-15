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
     * A low Mach number explosion: the density \f$\rho = 1 +
     * \varepsilon^2\f$ in the ball of radius \f$1/2\f$ and \f$\rho = 1\f$
     * outside, together with the radial momentum
     * \f[
     *   \boldsymbol m = -\max(0, 1-r)\,\big(1-e^{-16r^2}\big)\,
     *   \frac{\boldsymbol x}{r}.
     * \f]
     * The pressure (if the system has an energy equation) is set to
     * \f$\rho\f$.
     *
     * @note The @p t argument is ignored. This class always returns the
     * initial configuration.
     *
     * @ingroup EulerEquations
     */
    template <typename HyperbolicSystem, int dim, typename Number>
    class RadialExplosion : public InitialState<HyperbolicSystem, dim, Number>
    {
    public:
      using View = typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename View::state_type;

      RadialExplosion(const HyperbolicSystem &hyperbolic_system,
                      const std::string &subsection)
          : InitialState<HyperbolicSystem, dim, Number>("radial explosion",
                                                        subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
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

        const Number r = point.norm();
        const Number beta = std::max(Number(0.), Number(1.) - r) *
                            (Number(1.) - std::exp(Number(-16.) * r * r));

        state_type primitive_state;
        const Number rho =
            Number(1.) +
            (r <= Number(0.5) ? mach_number_ * mach_number_ : Number(0.));
        primitive_state[0] = rho;

        if (r > Number(0.))
          for (unsigned int d = 0; d < dim; ++d)
            primitive_state[1 + d] = -beta * point[d] / (rho * r);

        if constexpr (View::have_energy_equation)
          primitive_state[dim + 1] = rho;

        return view.from_primitive_state(primitive_state);
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;

      Number mach_number_;
    };
  } // namespace EulerInitialStates
} // namespace ryujin
