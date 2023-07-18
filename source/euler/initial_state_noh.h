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
    struct Description;

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
     * @ingroup EulerEquations
     */
    template <int dim, typename Number>
    class Noh : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystemView = HyperbolicSystem::View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      Noh(const HyperbolicSystemView &hyperbolic_system,
          const std::string &subsection)
          : InitialState<Description, dim, Number>("noh", subsection)
          , hyperbolic_system(hyperbolic_system)
      {
        /* No customization at this point */
      }

      /* Initial and exact solution for each dimension */
      auto compute(const dealii::Point<dim> &point, Number t)
          -> state_type final
      {
        // Initialize some quantities
        double rho = 1.;
        double u = 0., v = 0.;
        double E = 0.;
        const double gamma = hyperbolic_system.gamma();
        const auto norm = point.norm();

        // Define profiles for each dim here
        switch (dim) {
        case 1:

          /* Initial condition */
          if (t < 1.e-16) {
            if (norm > 1.e-16)
              u = -point[0] / norm;
            E = 1.e-12 / (gamma - Number(1.)) + Number(0.5) * rho * u * u;
          }

          /* Exact solution */
          else if (t / 3. < norm) {
            rho = 1.0;
            u = -point[0] / norm;
            E = 0.5 * rho + 1.e-12 / (gamma - Number(1.));
          } else if (t / 3. >= norm) {
            rho = 4.0;
            u = 0.0;
            E = 2.0 + 1.e-12 / (gamma - Number(1.));
          }

          break;
        case 2:

          /* Initial condition */
          if (t < 1.e-16) {
            if (norm > 1.e-16) {
              u = -point[0] / norm, v = -point[1] / norm;
            }
            E = 1.e-12 / (gamma - Number(1.)) +
                Number(0.5) * rho * (u * u + v * v);
          }

          /* Exact solution */
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

        /* Set final state */
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
      const HyperbolicSystemView hyperbolic_system;
    };

  } // namespace Euler
} // namespace ryujin
