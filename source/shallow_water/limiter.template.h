//
// SPDX-License-Identifier: MIT or BSD-3-Clause
// [LANL Copyright Statement]
// Copyright (C) 2020 - 2023 by the ryujin authors
// Copyright (C) 2023 - 2023 by Triad National Security, LLC
//

#pragma once

#include "limiter.h"

namespace ryujin
{
  namespace ShallowWater
  {
    template <int dim, typename Number>
    std::tuple<Number, bool>
    Limiter<dim, Number>::limit(const Bounds &bounds,
                                const state_type &U,
                                const state_type &P,
                                const Number t_min /* = Number(0.) */,
                                const Number t_max /* = Number(1.) */)
    {
      const auto &[h_min, h_max, h_small, kin_max, v2_max] = bounds;

      bool success = true;

      Number t_l = t_min;
      Number t_r = t_max;

      constexpr ScalarNumber min = std::numeric_limits<ScalarNumber>::min();
      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
      const auto small = hyperbolic_system.dry_state_relaxation_small();
      const auto large = hyperbolic_system.dry_state_relaxation_large();
      const auto relax_small = ScalarNumber(1. + small * eps);
      const auto relax = ScalarNumber(1. + large * eps);

      /*
       * We first limit the water_depth h.
       *
       * See [Guermond et al, 2021] (5.7).
       */

      {
        auto h_U = hyperbolic_system.water_depth(U);
        const auto &h_P = hyperbolic_system.water_depth(P);

        const auto test_min = hyperbolic_system.filter_dry_water_depth(
            std::max(Number(0.), h_U - relax * h_max));
        const auto test_max = hyperbolic_system.filter_dry_water_depth(
            std::max(Number(0.), h_min - relax * h_U));

        if (!(test_min == Number(0.) && test_max == Number(0.))) {
#ifdef DEBUG_OUTPUT
          std::cout << std::fixed << std::setprecision(16);
          std::cout << "Bounds violation: low-order water depth (critical)!\n"
                    << "\n\t\th min:         " << h_min
                    << "\n\t\th min (delta): " << negative_part(h_U - h_min)
                    << "\n\t\th:             " << h_U
                    << "\n\t\th max (delta): " << positive_part(h_U - h_max)
                    << "\n\t\th max:         " << h_max << "\n"
                    << std::endl;
#endif
          success = false;
        }

        const Number denominator =
            ScalarNumber(1.) / (std::abs(h_P) + eps * h_max + min);

        constexpr auto lt = dealii::SIMDComparison::less_than;

        t_r = dealii::compare_and_apply_mask<lt>( //
            h_max,
            h_U + t_r * h_P,
            /*
             * h_P is positive.
             *
             * Note: Do not take an absolute value here. If we are out of
             * bounds we have to ensure that t_r is set to t_min.
             */
            (h_max - h_U) * denominator,
            t_r);

        /*
         * Enforce strict limiting on dry states (i.e., if we happen to get
         * below h_small):
         */
        const auto h_min_tilde = std::max(h_small, h_min);

        t_r = dealii::compare_and_apply_mask<lt>( //
            h_U + t_r * h_P,
            h_min_tilde,
            /*
             * h_P is negative.
             *
             * Note: Do not take an absolute value here. If we are out of
             * bounds we have to ensure that t_r is set to t_min.
             */
            (h_U - h_min_tilde) * denominator,
            t_r);

        /*
         * Ensure that t_min <= t <= t_max. This might not be the case if
         * h_U is outside the interval [h_min, h_max]. Furthermore, the
         * quotient we take above is prone to numerical cancellation in
         * particular in the second pass of the limiter when h_P might be
         * small.
         */
        t_r = std::min(t_r, t_max);
        t_r = std::max(t_r, t_min);


#ifdef CHECK_BOUNDS
        /*
         * Verify that the new state is within bounds:
         */
        const auto h_new = hyperbolic_system.water_depth(U + t_r * P);
        const auto test_new_min = hyperbolic_system.filter_dry_water_depth(
            std::max(Number(0.), h_new - relax * h_max));
        const auto test_new_max = hyperbolic_system.filter_dry_water_depth(
            std::max(Number(0.), h_min - relax * h_new));

        if (!(test_new_min == Number(0.) && test_new_max == Number(0.))) {
#ifdef DEBUG_OUTPUT
          std::cout << std::fixed << std::setprecision(30);
          std::cout << "Bounds violation: high-order water depth!\n"
                    << "\n\t\th min:         " << h_min
                    << "\n\t\th min (delta): " << negative_part(h_new - h_min)
                    << "\n\t\th:             " << h_new
                    << "\n\t\th max (delta): " << positive_part(h_new - h_max)
                    << "\n\t\th max:         " << h_max << "\n"
                    << std::endl;
#endif
          success = false;
        }
#endif
      }

      /*
       * Limit the (negative) kinetic energy:
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

      if (hyperbolic_system.limiter_kinetic_energy()) {

        /* We first check if t_r is a good state */

        const auto U_r = U + t_r * P;
        const auto h_r = hyperbolic_system.water_depth(U_r);
        const auto q_r = hyperbolic_system.momentum(U_r);

        const auto psi_r =
            relax_small * h_r * kin_max - ScalarNumber(0.5) * q_r.norm_square();

        /*
         * If psi_r > 0 the right state is fine, force returning t_r by
         * setting t_l = t_r:
         */
        t_l = dealii::compare_and_apply_mask<
            dealii::SIMDComparison::greater_than>(psi_r, Number(0.), t_r, t_l);

        /* If we have set t_l = t_r everywhere we can return: */
        if (!hyperbolic_system.limiter_square_velocity() && t_l == t_r)
          return {t_l, success};

#ifdef DEBUG_OUTPUT_LIMITER
        {
          std::cout << std::endl;
          std::cout << std::fixed << std::setprecision(16);
          std::cout << "t_l: (start) " << t_l << std::endl;
          std::cout << "t_r: (start) " << t_r << std::endl;
        }
#endif

        const auto U_l = U + t_l * P;
        const auto h_l = hyperbolic_system.water_depth(U_l);
        const auto q_l = hyperbolic_system.momentum(U_l);

        const auto psi_l =
            relax_small * h_l * kin_max - ScalarNumber(0.5) * q_l.norm_square();

        /*
         * Verify that the left state is within bounds. This property might
         * be violated for relative CFL numbers larger than 1.
         *
         * We use a non-scaled "+min" here to force the lower_bound to be
         * negative so that we do not accidentally trigger in "perfect" dry
         * states with h_l equal to zero.
         */
        const auto filtered_h_l = hyperbolic_system.filter_dry_water_depth(h_l);
        const auto lower_bound =
            (ScalarNumber(1.) - relax) * filtered_h_l * kin_max - eps;
        if (!(std::min(Number(0.), psi_l - lower_bound) == Number(0.))) {
#ifdef DEBUG_OUTPUT
          std::cout << std::fixed << std::setprecision(16);
          std::cout
              << "Bounds violation: low-order kinetic energy (critical)!\n";
          std::cout << "\t\tPsi left: 0 <= " << psi_l << "\n" << std::endl;
#endif
          success = false;
        }

        /*
         * Skip the quadratic Newton step if the window between t_l and t_r
         * is within the prescribed tolerance:
         */
        if (!(std::max(Number(0.), t_r - t_l - newton_tol) == Number(0.))) {
          /*
           * If the bound is not satisfied, we need to find the root of a
           * quadratic function:
           *
           * psi(t)   = (h_U + t h_P) kin_max
           *            - 1/2 (|q_U|^2 + 2(q_U * q_P) t + |q_P|^2 t^2)
           *
           * d_psi(t) = h_P kin_max - (q_U * q_P) - |q_P|^2 t
           *
           * We can compute the root of this function efficiently by using our
           * standard quadratic_newton_step() function that will use the points
           * [p1, p1, p2] as well as [p1, p2, p2] to construct two quadratic
           * polynomials to compute new candiates for the bounds [t_l, t_r]. In
           * case of a quadratic function psi(t) both polynomials will coincide
           * so that (up to round-off error) t_l = t_r.
           */

          const auto &h_P = hyperbolic_system.water_depth(P);
          const auto &q_U = hyperbolic_system.momentum(U);
          const auto &q_P = hyperbolic_system.momentum(P);

          const auto dpsi_l = h_P * kin_max - (q_U * q_P) - q_P * q_P * t_l;
          const auto dpsi_r = h_P * kin_max - (q_U * q_P) - q_P * q_P * t_r;

          quadratic_newton_step(
              t_l, t_r, psi_l, psi_r, dpsi_l, dpsi_r, Number(-1.));

#ifdef DEBUG_OUTPUT_LIMITER
          if (std::max(Number(0.), psi_r + Number(eps)) == Number(0.)) {
            std::cout << "psi_l:       " << psi_l << std::endl;
            std::cout << "psi_r:       " << psi_r << std::endl;
            std::cout << "dpsi_l:      " << dpsi_l << std::endl;
            std::cout << "dpsi_r:      " << dpsi_r << std::endl;
            std::cout << "t_l: (end)   " << t_l << std::endl;
            std::cout << "t_r: (end)   " << t_r << std::endl;
          }
#endif
        }

#ifdef CHECK_BOUNDS
        /*
         * Verify that the new state is within bounds:
         */
        {
          const auto U_new = U + t_l * P;
          const auto h_new = hyperbolic_system.water_depth(U_new);
          const auto q_new = hyperbolic_system.momentum(U_new);

          const auto psi_new = relax_small * h_new * kin_max -
                               ScalarNumber(0.5) * q_new.norm_square();

          const auto lower_bound =
              (ScalarNumber(1.) - relax) * h_new * kin_max - eps;

          const bool psi_valid =
              std::min(Number(0.), psi_new - lower_bound) == Number(0.);
          if (!psi_valid) {
#ifdef DEBUG_OUTPUT
            std::cout << std::fixed << std::setprecision(16);
            std::cout << "Bounds violation: high-order kinetic energy!\n";
            std::cout << "\t\tPsi: 0 <= " << psi_new << "\n" << std::endl;
#endif
            success = false;
          }
        }
#endif

        /* Flip bounds for the square velocity limiter: */
        if (hyperbolic_system.limiter_square_velocity()) {
          t_r = t_l;
          t_l = t_min;
        }
      }

      /*
       * Limit the (negative) |v|^2:
       *
       * Given initial limiter values t_l and t_r with psi(t_l) > 0 and
       * psi(t_r) < 0 we try to find t^\ast with psi(t^\ast) \approx 0.
       *
       * Here, psi is the function:
       *
       *   psi = h^2 (|v|^2)^max - |q|^2
       */

      if (hyperbolic_system.limiter_square_velocity()) {
        /* We first check if t_r is a good state */

        const auto U_r = U + t_r * P;
        const auto h_r = hyperbolic_system.water_depth(U_r);
        const auto q_r = hyperbolic_system.momentum(U_r);

        const auto psi_r = relax_small * h_r * h_r * v2_max - q_r.norm_square();

        /*
         * If psi_r > 0 the right state is fine, force returning t_r by
         * setting t_l = t_r:
         */
        t_l = dealii::compare_and_apply_mask<
            dealii::SIMDComparison::greater_than>(psi_r, Number(0.), t_r, t_l);

        /* If we have set t_l = t_r everywhere we can return: */
        if (t_l == t_r)
          return {t_l, success};

#ifdef DEBUG_OUTPUT_LIMITER
        {
          std::cout << std::endl;
          std::cout << std::fixed << std::setprecision(16);
          std::cout << "t_l: (start) " << t_l << std::endl;
          std::cout << "t_r: (start) " << t_r << std::endl;
        }
#endif

        const auto U_l = U + t_l * P;
        const auto h_l = hyperbolic_system.water_depth(U_l);
        const auto q_l = hyperbolic_system.momentum(U_l);

        const auto psi_l = relax_small * h_l * h_l * v2_max - q_l.norm_square();

        /*
         * Verify that the left state is within bounds. This property might
         * be violated for relative CFL numbers larger than 1.
         *
         * We use a non-scaled eps here to force the lower_bound to be
         * negative so that we do not accidentally trigger in "perfect" dry
         * states with h_l equal to zero.
         */
        const auto filtered_h_l = hyperbolic_system.filter_dry_water_depth(h_l);
        const auto lower_bound =
            (ScalarNumber(1.) - relax) * filtered_h_l * filtered_h_l * v2_max -
            ScalarNumber(100.) * eps;
        if (!(std::min(Number(0.), psi_l - lower_bound) == Number(0.))) {
#ifdef DEBUG_OUTPUT
          std::cout << std::fixed << std::setprecision(16);
          std::cout
              << "Bounds violation: low-order square velocity (critical)!\n";
          std::cout << "\t\tPsi left: 0 <= " << psi_l << "\n" << std::endl;
#endif
          success = false;
        }

        /*
         * Skip the quadratic Newton step if the window between t_l and t_r
         * is within the prescribed tolerance:
         */
        if (!(std::max(Number(0.), t_r - t_l - newton_tol) == Number(0.))) {
          /*
           * If the bound is not satisfied, we need to find the root of a
           * quadratic function:
           *
           * psi(t)   = (h_U + t h_P)^2 v2_max
           *            - (|q_U|^2 + 2(q_U * q_P) t + |q_P|^2 t^2)
           *
           * d_psi(t) = 2 (h_U + t * h_P) * h_P v2_max
           *            - 2 (q_U * q_P) - |q_P|^2 t
           *
           * We can compute the root of this function efficiently by using our
           * standard quadratic_newton_step() function that will use the points
           * [p1, p1, p2] as well as [p1, p2, p2] to construct two quadratic
           * polynomials to compute new candiates for the bounds [t_l, t_r]. In
           * case of a quadratic function psi(t) both polynomials will coincide
           * so that (up to round-off error) t_l = t_r.
           */
          const auto &h_U = hyperbolic_system.water_depth(U);
          const auto &h_P = hyperbolic_system.water_depth(P);
          const auto &q_U = hyperbolic_system.momentum(U);
          const auto &q_P = hyperbolic_system.momentum(P);

          const auto dpsi_l =
              (h_U + t_l * h_P) * h_P * v2_max -
              ScalarNumber(2.) * ((q_U * q_P) - q_P * q_P * t_l);
          const auto dpsi_r =
              (h_U + t_r * h_P) * h_P * v2_max -
              ScalarNumber(2.) * ((q_U * q_P) - q_P * q_P * t_r);

          quadratic_newton_step(
              t_l, t_r, psi_l, psi_r, dpsi_l, dpsi_r, Number(-1.));

#ifdef DEBUG_OUTPUT_LIMITER
          if (std::max(Number(0.), psi_r + Number(eps)) == Number(0.)) {
            std::cout << "psi_l:       " << psi_l << std::endl;
            std::cout << "psi_r:       " << psi_r << std::endl;
            std::cout << "dpsi_l:      " << dpsi_l << std::endl;
            std::cout << "dpsi_r:      " << dpsi_r << std::endl;
            std::cout << "t_l: (end)   " << t_l << std::endl;
            std::cout << "t_r: (end)   " << t_r << std::endl;
          }
#endif
        }

#ifdef CHECK_BOUNDS
        /*
         * Verify that the new state is within bounds:
         */
        {
          const auto U_new = U + t_l * P;
          const auto h_new = hyperbolic_system.water_depth(U_new);
          const auto q_new = hyperbolic_system.momentum(U_new);

          const auto psi_new =
              relax_small * h_new * h_new * v2_max - q_new.norm_square();

          const auto lower_bound =
              (ScalarNumber(1.) - relax) * h_new * h_new * v2_max -
              ScalarNumber(100.) * eps;

          const bool psi_valid =
              std::min(Number(0.), psi_new - lower_bound) == Number(0.);
          if (!psi_valid) {
#ifdef DEBUG_OUTPUT
            std::cout << std::fixed << std::setprecision(16);
            std::cout << "Bounds violation: high-order square velocity!\n";
            std::cout << "\t\tPsi: 0 <= " << psi_new << "\n" << std::endl;
#endif
            success = false;
          }
        }
#endif
      }

      return {t_l, success};
    }

  } // namespace ShallowWater
} // namespace ryujin
