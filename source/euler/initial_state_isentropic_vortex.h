//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"
#include <initial_state.h>

namespace ryujin
{
  namespace Euler
  {
    /**
     * The isentropic vortex
     * @todo Documentation
     *
     * @ingroup EulerEquations
     */
    template <int dim, typename Number, typename state_type>
    class IsentropicVortex : public InitialState<dim, Number, state_type>
    {
    public:
      using HyperbolicSystemView = HyperbolicSystem::View<dim, Number>;

      IsentropicVortex(const HyperbolicSystemView &hyperbolic_system,
                       const std::string subsection)
          : InitialState<dim, Number, state_type>("isentropic vortex",
                                                  subsection)
          , hyperbolic_system(hyperbolic_system)
      {
        mach_number_ = 2.0;
        this->add_parameter(
            "mach number", mach_number_, "Mach number of isentropic vortex");

        beta_ = 5.0;
        this->add_parameter("beta", beta_, "vortex strength beta");
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        const auto gamma = hyperbolic_system.gamma();


        /* In 3D we simply project onto the 2d plane: */
        dealii::Point<2> point_bar;
        point_bar[0] = point[0] - mach_number_ * t;
        point_bar[1] = point[1];

        const Number r_square = Number(point_bar.norm_square());

        const Number factor = beta_ / Number(2. * M_PI) *
                              exp(Number(0.5) - Number(0.5) * r_square);

        const Number T = Number(1.) - (gamma - Number(1.)) /
                                          (Number(2.) * gamma) * factor *
                                          factor;

        const Number u = mach_number_ - factor * Number(point_bar[1]);
        const Number v = factor * Number(point_bar[0]);

        const Number rho = ryujin::pow(T, Number(1.) / (Number(gamma - 1.)));
        const Number p = ryujin::pow(rho, Number(gamma));
        const Number E =
            p / (gamma - Number(1.)) + Number(0.5) * rho * (u * u + v * v);

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
      const HyperbolicSystemView &hyperbolic_system;

      Number mach_number_;
      Number beta_;
    };
  } // namespace Euler
} // namespace ryujin
