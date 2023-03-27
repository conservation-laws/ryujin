//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <hyperbolic_system.h>
#include <initial_state.h>

namespace ryujin
{
  namespace Euler
  {
    /**
     *
     * This initial state reproduces the classical Noh problem introduced
     * in:
     *
     * W.F Noh, Errors for calculations of strong shocks using an artificial
     * viscosity and an artificial heat flux, Journal of Computational
     * Physics, Volume 72, Issue 1, 1987,
     * https://doi.org/10.1016/0021-9991(87)90074-X.
     *
     * @ingroup InitialValues
     */
    template <int dim, typename Number, typename state_type>
    class Noh : public InitialState<dim, Number, state_type>
    {
    public:
      Noh(const HyperbolicSystem &hyperbolic_system,
          const std::string subsection)
          : InitialState<dim, Number, state_type>("noh", subsection)
          , hyperbolic_system(hyperbolic_system)
      {
        /* no customization at this point */
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        // initialize some values
        auto rho = 1.;
        auto u = 0., v = 0.;
        auto E = 0.;

        const auto gamma = hyperbolic_system.gamma();
        const auto norm = point.norm();

        // initial/exact solution for each dimension
        switch (dim) {
        case 1:

          // initial condition
          if (t < 1.e-16) {
            if (norm > 1.e-16)
              u = -point[0] / norm;
            E = 1.e-12 / (gamma - Number(1.)) + Number(0.5) * rho * u * u;
          }

          // exact solution
          else if (t / 3. < norm) {
            rho = 1.0;
            u = -1;
            E = 0.5 * rho + 1.e-12 / (gamma - Number(1.));
          } else if (t / 3. >= norm) {
            rho = 4.0;
            u = 0.0;
            E = 2.0 + 1.e-12 / (gamma - Number(1.));
          }

          break;
        case 2:

          // initial condition
          if (t < 1.e-16) {
            if (norm > 1.e-16) {
              u = -point[0] / norm, v = -point[1] / norm;
            }
            E = 1.e-12 / (gamma - Number(1.)) +
                Number(0.5) * rho * (u * u + v * v);
          }

          // exact solution
          else if (t / 3. < norm) {
            rho = 1.0 + t / norm;
            u = -point[0] / norm, v = -point[1] / norm;
            E = 0.5 * rho + 1.e-12 / (gamma - Number(1.));
          } else if (t / 3. >= norm) {
            rho = 16.0;
            u = 0.0, v = 0.0;
            E = 8.0 + 1.e-12 / (gamma - Number(1.));
          }

          break;
        }

        // Set final state
        if constexpr (dim == 1)
          return state_type({rho, rho * u, E});
        else if constexpr (dim == 2)
          return state_type({rho, rho * u, rho * v, E});
        else {
          AssertThrow(false, dealii::ExcNotImplemented());
          __builtin_trap();
        }
      }

    private:
      const HyperbolicSystem &hyperbolic_system;
    };

  } // namespace Euler
} // namespace ryujin
