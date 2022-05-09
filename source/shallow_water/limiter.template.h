//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "limiter.h"

namespace ryujin
{
  template <int dim, typename Number>
  std::tuple<Number, bool>
  Limiter<dim, Number>::limit(const HyperbolicSystem &hyperbolic_system,
                              const Bounds &bounds,
                              const state_type &U,
                              const state_type &P,
                              const ScalarNumber /* newton_tolerance */,
                              const unsigned int /* newton_max_iter */,
                              const Number t_min /* = Number(0.) */,
                              const Number t_max /* = Number(1.) */)
  {
    bool success = true;
    Number t_r = t_max;

    constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
    // FIXME
    // constexpr ScalarNumber relax = ScalarNumber(1. + 10. * eps);
    constexpr ScalarNumber relaxbig = ScalarNumber(1. + 10000. * eps);

    /*
     * We first limit the water_depth h.
     *
     * See [Guermond et al, 2021] (5.7).
     */

    {
      const auto &U_h = hyperbolic_system.water_depth(U);
      const auto &P_h = hyperbolic_system.water_depth(P);

      const auto &h_min = std::get<0>(bounds);
      const auto &h_max = std::get<1>(bounds);
      const auto h_tiny = hyperbolic_system.h_tiny();

      if (!((std::max(Number(0.), U_h - relaxbig * h_max) == Number(0.)) &&
            (std::max(Number(0.), h_min - relaxbig * U_h) == Number(0.)))) {
#ifdef DEBUG_OUTPUT
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Bounds violation: low-order water depth (critical)!\n";
        std::cout << "\t\th min: " << h_min << "\n";
        std::cout << "\t\th:     " << U_h << "\n";
        std::cout << "\t\th max: " << h_max << "\n" << std::endl;
#endif
        success = false;
      }

      const Number denominator =
          ScalarNumber(1.) / (std::abs(P_h) + eps * h_max);
      t_r = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
          h_max,
          U_h + t_r * P_h,
          (std::abs(h_max - U_h) + eps * h_min) * denominator,
          t_r);

      t_r = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
          U_h + t_r * P_h,
          h_min,
          (std::abs(h_min - U_h) + eps * h_min) * denominator,
          t_r);

      /* Check if h <= h_tiny force low-order: */
      // FIXME verify
      t_r = dealii::compare_and_apply_mask<
          dealii::SIMDComparison::less_than_or_equal>(
          U_h, h_tiny, Number(0.), t_r);

      /*
       * It is always t_min <= t_r <= t_max, but just to be sure, box
       * back into bounds:
       */
      t_r = std::min(t_r, t_max);
      t_r = std::max(t_r, t_min);

#ifdef CHECK_BOUNDS
      /*
       * Verify that the new state is within bounds:
       */
      const auto n_h = hyperbolic_system.water_depth(U + t_r * P);
      if (!((std::max(Number(0.), n_h - relaxbig * h_max) == Number(0.)) &&
            (std::max(Number(0.), h_min - relaxbig * n_h) == Number(0.)))) {
#ifdef DEBUG_OUTPUT
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Bounds violation: high-order water depth!\n";
        std::cout << "\t\th min: " << h_min << "\n";
        std::cout << "\t\th:     " << n_h << "\n";
        std::cout << "\t\th max: " << h_max << "\n" << std::endl;
#endif
        success = false;
      }
#endif
    }


    /*
     * Then limit the (negative) kinetic energy:
     *
     * See [Guermond et al, 2021] (5.10).
     *
     * Given initial limiter values t_l and t_r with psi(t_l) > 0 and
     * psi(t_r) < 0 we try to find t^\ast with psi(t^\ast) \approx 0.
     *
     * Here, psi is the function:
     *
     *   psi = h KE_i^max - 1/2 |q|^2
     */

    Number t_l = t_min; // good state

    {
      const auto &U_h = hyperbolic_system.water_depth(U);
      const auto &P_h = hyperbolic_system.water_depth(P);

      const auto &U_m = hyperbolic_system.momentum(U);
      const auto &P_m = hyperbolic_system.momentum(P);

      const auto &kin_max = std::get<2>(bounds);
      const Number h_kin_small = hyperbolic_system.h_kinetic_energy_tiny();

      /* We first check if t_r is a good state */

      const auto U_r = U + t_r * P;
      const auto h_r = hyperbolic_system.water_depth(U_r);
      const auto q_r = hyperbolic_system.momentum(U_r);

      const auto psi_r = h_r * kin_max - ScalarNumber(0.5) * q_r.norm_square();

      /* If psi_r > -h_kin_small the right state is fine, force returning t_r by
       * setting t_l = t_r: */
      t_l = dealii::compare_and_apply_mask< //
          dealii::SIMDComparison::greater_than>(psi_r, -h_kin_small, t_r, t_l);
      // FIXME

      /* If we have set t_l = t_r everywhere we can return: */
      if (t_l == t_r)
        return {t_l, success};

      /*
       * If bound is not satisfied, we need to find the root of a quadratic
       * equation:
       *
       * a t^2 + b t + c = 0
       *
       * If root r exists, we take t_l = min(r, t_r). Else we take t_l =
       * min(t_l, t_r). Up to round-off error, a < 0 and c >= 0 always, so
       * we only define the negative root of the quadratic equation.
       */

      const auto a = -ScalarNumber(0.5) * P_m.norm_square();
      const auto a_nudged = std::min(a, -h_kin_small); // FIXME
      const auto b = P_h * kin_max - U_m * P_m;
      const auto c = U_h * kin_max - ScalarNumber(0.5) * U_m.norm_square();

      // FIXME: Check admissibility

      const Number discriminant = b * b - ScalarNumber(4.) * a * c;

      Number root = ScalarNumber(0.5) / a_nudged *
                    (-b - std::sqrt(std::abs(discriminant)));

      /* Define final limiter */
      t_l =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
              root, Number(0.), std::min(root, t_r), std::min(t_l, t_r));

      /* If kinetic energy <= h_kin_small, then set limiter to 0 */
      t_l = dealii::compare_and_apply_mask<
          dealii::SIMDComparison::less_than_or_equal>(
          U_h * kin_max, h_kin_small, Number(0.), t_l);
      // FIXME

      /* and then put it in a box for safety reasons */
      t_l = std::min(t_l, t_max);
      t_l = std::max(t_l, t_min);
    }

    return {t_l, success};
  }

} // namespace ryujin
