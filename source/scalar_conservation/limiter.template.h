//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2024 by the ryujin authors
//

#pragma once

#include "limiter.h"

namespace ryujin
{
  namespace ScalarConservation
  {
    template <int dim, typename Number>
    std::tuple<Number, bool>
    Limiter<dim, Number>::limit(const Bounds &bounds,
                                const state_type &U,
                                const state_type &P,
                                const Number t_min /* = Number(0.) */,
                                const Number t_max /* = Number(1.) */)
    {
      const auto view = hyperbolic_system.view<dim, Number>();

      bool success = true;
      Number t_r = t_max;

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
      const ScalarNumber relax = ScalarNumber(1. + 10000. * eps);

      const auto &u_U = view.state(U);
      const auto &u_P = view.state(P);

      const auto &u_min = std::get<0>(bounds);
      const auto &u_max = std::get<1>(bounds);

      /*
       * Verify that u_U is within bounds. This property might be
       * violated for relative CFL numbers larger than 1.
       *
       * u_min, u_U, u_max might be negative, thus relax in both directions.
       */
      const auto test_max = std::max(
          Number(0.), std::min(u_U - relax * u_max, relax * u_U - u_max));
      const auto test_min = std::max(
          Number(0.), std::min(u_min - relax * u_U, relax * u_min - u_U));
      if (!(test_max == Number(0.) && test_min == Number(0.))) {
#ifdef DEBUG_OUTPUT
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Bounds violation: low-order state (critical)!"
                  << "\n\t\tu min:         " << u_min
                  << "\n\t\tu min (delta): " << negative_part(u_U - u_min)
                  << "\n\t\tu:             " << u_U
                  << "\n\t\tu max (delta): " << positive_part(u_U - u_max)
                  << "\n\t\tu max:         " << u_max << "\n"
                  << std::endl;
#endif
        success = false;
      }

      const auto regularization =
          Number(100. * std::numeric_limits<ScalarNumber>::min());

      const Number denominator =
          ScalarNumber(1.) /
          std::max(regularization, std::abs(u_P) + eps * u_max);

      t_r = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
          u_max,
          u_U + t_r * u_P,
          /*
           * u_P is positive.
           *
           * Note: Do not take an absolute value here. If we are out of
           * bounds we have to ensure that t_r is set to t_min.
           */
          (u_max - u_U) * denominator,
          t_r);

      t_r = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
          u_U + t_r * u_P,
          u_min,
          /*
           * u_P is negative.
           *
           * Note: Do not take an absolute value here. If we are out of
           * bounds we have to ensure that t_r is set to t_min.
           */
          (u_U - u_min) * denominator,
          t_r);

      /*
       * Ensure that t_min <= t <= t_max. This might not be the case if
       * u_U is outside the interval [u_min, u_max]. Furthermore,
       * the quotient we take above is prone to numerical cancellation in
       * particular in the second pass of the limiter when u_P might be
       * small.
       */
      t_r = std::min(t_r, t_max);
      t_r = std::max(t_r, t_min);

#ifdef EXPENSIVE_BOUNDS_CHECK
      /*
       * Verify that the new state is within bounds:
       *
       * u_min, u_U, u_max might be negative, thus relax in both directions.
       */
      const auto u_new = view.state(U + t_r * P);
      const auto test_new_max = std::max(
          Number(0.), std::min(u_new - relax * u_max, relax * u_new - u_max));
      const auto test_new_min = std::max(
          Number(0.), std::min(u_min - relax * u_new, relax * u_min - u_new));
      if (!(test_new_max == Number(0.) && test_new_min == Number(0.))) {
#ifdef DEBUG_OUTPUT
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Bounds violation: high-order state!"
                  << "\n\t\tu min:         " << u_min
                  << "\n\t\tu min (delta): " << negative_part(u_new - u_min)
                  << "\n\t\tu:             " << u_new
                  << "\n\t\tu max (delta): " << positive_part(u_new - u_max)
                  << "\n\t\tu max:         " << u_max << "\n"
                  << std::endl;
#endif
        success = false;
      }
#endif

      return {t_r, success};
    }

  } // namespace ScalarConservation
} // namespace ryujin
