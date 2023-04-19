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
     * The rarefaction
     * @todo Documentation
     *
     * @ingroup InitialValues
     */

    template <int dim, typename Number, typename state_type>
    class Rarefaction : public InitialState<dim, Number, state_type>
    {
    public:
      Rarefaction(const HyperbolicSystem &hyperbolic_system,
                  const std::string subsection)
          : InitialState<dim, Number, state_type>("rarefaction", subsection)
          , hyperbolic_system(hyperbolic_system)
      {
      } /* Constructor */

      state_type compute(const dealii::Point<dim> &point, Number t) final;

    private:
      const HyperbolicSystem &hyperbolic_system;
    }; // Rarefaction

#if 0
//       Number cL, rhoL, uL, pL;
//       Number rhoR, uR, pR, cR;
//       Number k1, k2, dens_pow, k3, p_pow;
#endif

    template <int dim, typename Number, typename state_type>
    state_type Rarefaction<dim, Number, state_type>::compute(
        const dealii::Point<dim> &point, Number t)
    {
      const Number gamma = hyperbolic_system.gamma();

      using state_type_1d = std::array<Number, 3>;

      /*
       * Compute the speed of sound:
       */
      const auto speed_of_sound = [&](const auto rho, const auto p) {
        return std::sqrt(gamma * p / rho);
      };

      /*
       * Compute the rarefaction right side:
       */
      const auto rarefaction_right_state = [&](const auto primitive_left,
                                               const auto rho_right) {
        const auto &[rho_left, u_left, p_left] = primitive_left;
        state_type_1d primitive_right{{rho_right, 0., 0.}};

        /* Isentropic condition: pR = (rhoR/rhoL)^{gamma} * pL */
        primitive_right[2] = std::pow(rho_right / rho_left, gamma) * p_left;

        const auto c_left = speed_of_sound(rho_left, p_left);
        const auto c_right = speed_of_sound(rho_right, primitive_right[2]);

        /* 1-Riemann invariant: uR + 2 cR/(gamma -1) = uL + 2 cL/(gamma -1) */
        primitive_right[1] =
            u_left + 2.0 * (c_left - c_right) / (gamma - 1.0);

        return primitive_right;
      };

      /* Initial left and right states (rho, u, p): */
      const state_type_1d primitive_left{{3.0, speed_of_sound(3.0, 1.0), 1.0}};
      const state_type_1d primitive_right =
          rarefaction_right_state(primitive_left, /*rho_right*/ 0.5);

      /*
       * Compute rarefaction solution:
       */

      const auto &[rho_left, u_left, p_left] = primitive_left;
      const auto c_left = speed_of_sound(rho_left, p_left);
      const auto &[rho_right, u_right, p_right] = primitive_right;
      const auto c_right = speed_of_sound(rho_right, p_right);

      /* Constants: */
      const Number k1 = 2.0 / (gamma + 1.0);
      const Number k2 = ((gamma - 1.0) / ((gamma + 1.0) * c_left));
      const Number density_exponent = 2.0 / (gamma - 1.0);
      const Number k3 = c_left + ((gamma - 1.0) / 2.0) * u_left;
      const Number pressure_exponent = 2.0 * gamma / (gamma - 1.0);

      const double &x = point[0];

      state_type_1d primitive;

      if (x <= t * (u_left - c_left)) {
        primitive = primitive_left;

      } else if (x <= t * (u_right - c_right)) {

        /* Self-similar variable: */
        const double chi = x / t;

        primitive[0] =
            rho_left * std::pow(k1 + k2 * (u_left - chi), density_exponent);
        primitive[1] = k1 * (k3 + chi);
        primitive[2] =
            p_left * std::pow(k1 + k2 * (u_left - chi), pressure_exponent);

      } else {
        primitive = primitive_right;
      }

      return hyperbolic_system.template expand_state<dim>(
          hyperbolic_system.from_primitive_state(dealii::Tensor<1, 3, Number>{
              {primitive[0], primitive[1], primitive[2]}}));
    }

  } // namespace Euler
} // namespace ryujin
