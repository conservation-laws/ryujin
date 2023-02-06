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
     * We recommend Sect. 6.7 of the following article for more details:
     *
     * Guermond, JL., Popov, B. & Saavedra, L. Second-Order Invariant Domain
     * Preserving ALE Approximation of Euler Equations. Commun. Appl. Math.
     * Comput. (2021). https://doi.org/10.1007/s42967-021-00165-y
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
        // no customization at this point
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        const auto gamma = hyperbolic_system.gamma();

        // initialize some things
        auto rho = 1.0;
        auto u = 0.0;
        auto v = 0.0;
        auto E = 0.0;

        const auto norm = point.norm();

        if (t < 1.e-16) {
          if (norm > 1.e-16) {
            u = -point[0] / norm;
            v = -point[1] / norm;
          }
          // Get E from ideal EOS
          E = 1.e-12 / (gamma - Number(1.)) +
              Number(0.5) * rho * (u * u + v * v);
        } else if (t / 3. < point.norm()) {
          rho = 1.0 + t / norm;
          u = 0.0;
          v = 0.0;
          E = 0.5 * rho;
        } else {
          rho = 16.0;
          if (norm > 1.e-16) {
            u = -point[0] / norm;
            v = -point[1] / norm;
          }
          E = 8.0;
        }


        // Set final state
        if constexpr (dim == 1)
          return state_type({rho, rho * u, E});
        else if constexpr (dim == 2)
          return state_type({rho, rho * u, rho * v, E});
        else if constexpr (dim == 3)
          return state_type({rho, rho * u, rho * v, Number(0.), E});
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
