//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <initial_state_library.h>
#include <simd.h>

namespace ryujin
{
  namespace Euler
  {
    struct Description;
  }

  namespace EulerInitialStates
  {
    /**
     * The isentropic vortex
     * @todo Documentation
     *
     * @ingroup EulerEquations
     */
    template <typename Description, int dim, typename Number>
    class IsentropicVortex : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using HyperbolicSystemView =
          typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      IsentropicVortex(const HyperbolicSystemView &hyperbolic_system,
                       const std::string subsection)
          : InitialState<Description, dim, Number>("isentropic vortex",
                                                   subsection)
          , hyperbolic_system(hyperbolic_system)
      {
        gamma_ = 1.4;
        if constexpr (!std::is_same_v<Description, Euler::Description>) {
          this->add_parameter("gamma", gamma_, "The ratio of specific heats");
        }

        mach_number_ = 2.0;
        this->add_parameter(
            "mach number", mach_number_, "Mach number of isentropic vortex");

        beta_ = 5.0;
        this->add_parameter("beta", beta_, "vortex strength beta");
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        if constexpr (std::is_same_v<Description, Euler::Description>) {
          gamma_ = this->hyperbolic_system.gamma();
        }

        /* In 3D we simply project onto the 2d plane: */
        dealii::Point<2> point_bar;
        point_bar[0] = point[0] - mach_number_ * t;
        point_bar[1] = point[1];

        const Number r_square = Number(point_bar.norm_square());

        const Number factor = beta_ / Number(2. * M_PI) *
                              exp(Number(0.5) - Number(0.5) * r_square);

        const Number T = Number(1.) - (gamma_ - Number(1.)) /
                                          (Number(2.) * gamma_) * factor *
                                          factor;

        const Number u = mach_number_ - factor * Number(point_bar[1]);
        const Number v = factor * Number(point_bar[0]);

        const Number rho = ryujin::pow(T, Number(1.) / (Number(gamma_ - 1.)));
        const Number p = ryujin::pow(rho, Number(gamma_));
        const Number E =
            p / (gamma_ - Number(1.)) + Number(0.5) * rho * (u * u + v * v);

        if constexpr (dim == 2)
          return state_type({rho, rho * u, rho * v, E});
        else if constexpr (dim == 3)
          return state_type({rho, rho * u, rho * v, Number(0.), E});
        else {
          AssertThrow(false, dealii::ExcNotImplemented());
          __builtin_trap();
        }
      }

    private:
      const HyperbolicSystemView hyperbolic_system;

      Number gamma_;
      Number mach_number_;
      Number beta_;
    };
  } // namespace Euler
} // namespace ryujin
