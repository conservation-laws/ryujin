//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include <initial_state_library.h>

namespace ryujin
{
  namespace Euler
  {
    struct Description;
  }

  namespace EulerInitialStates
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
     * @ingroup EulerEquations
     */
    template <typename Description, int dim, typename Number>
    class Noh : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using HyperbolicSystemView =
          typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      Noh(const HyperbolicSystem &hyperbolic_system,
          const std::string &subsection)
          : InitialState<Description, dim, Number>("noh", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        gamma_ = 1.4;
        if constexpr (!std::is_same_v<Description, Euler::Description>) {
          this->add_parameter("gamma", gamma_, "The ratio of specific heats");
        }

        this->parse_parameters_call_back.connect([&]() {
          if constexpr (std::is_same_v<Description, Euler::Description>) {
            gamma_ = hyperbolic_system_.gamma();
          }
        });
      }

      /* Initial and exact solution for each dimension */
      auto compute(const dealii::Point<dim> &point, Number t)
          -> state_type final
      {
        // Initialize some quantities
        Number rho = 1.;
        Number u = 0.;
        Number v = 0.;
        Number E = 0.;

        const auto norm = point.norm();

        // Define profiles for each dim here
        switch (dim) {
        case 1:
          if (t < 1.e-16) {
            /* Initial condition */
            if (norm > 1.e-16)
              u = -point[0] / norm;
            E = 1.e-12 / (gamma_ - Number(1.)) + Number(0.5) * rho * u * u;
          } else if (t / 3. < norm) {
            /* Exact solution */
            rho = 1.0;
            u = -point[0] / norm;
            E = 0.5 * rho + 1.e-12 / (gamma_ - Number(1.));
          } else if (t / 3. >= norm) {
            rho = 4.0;
            u = 0.0;
            E = 2.0 + 1.e-12 / (gamma_ - Number(1.));
          }
          break;

        case 2:
          if (t < 1.e-16) {
            /* Initial condition */
            if (norm > 1.e-16) {
              u = -point[0] / norm, v = -point[1] / norm;
            }
            E = 1.e-12 / (gamma_ - Number(1.)) +
                Number(0.5) * rho * (u * u + v * v);

          } else if (t / 3. < norm) {
            /* Exact solution */
            rho = 1.0 + t / norm;
            u = -point[0] / norm, v = -point[1] / norm;
            E = 0.5 * rho + 1.e-12 / (gamma_ - Number(1.));

          } else if (t / 3. >= norm) {
            rho = 16.0;
            u = 0.0, v = 0.0;
            E = 8.0 + 1.e-12 / (gamma_ - Number(1.));
          }
          break;

        case 3:
          AssertThrow(false, dealii::ExcNotImplemented());
          __builtin_trap();
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
      const HyperbolicSystemView hyperbolic_system_;
      Number gamma_;
    };
  } // namespace EulerInitialStates
} // namespace ryujin
