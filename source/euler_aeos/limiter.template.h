//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include "limiter.h"
// #define DEBUG_OUTPUT_LIMITER

namespace ryujin
{
  namespace EulerAEOS
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
      const auto small = view.vacuum_state_relaxation_small();
      const auto large = view.vacuum_state_relaxation_large();
      const ScalarNumber relax_small = ScalarNumber(1. + small * eps);
      const ScalarNumber relax = ScalarNumber(1. + large * eps);

      /*
       * First limit the density rho.
       *
       * See [Guermond, Nazarov, Popov, Thomas] (4.8):
       */

      {
        const auto &rho_U = view.density(U);
        const auto &rho_P = view.density(P);

        const auto &rho_min = std::get<0>(bounds);
        const auto &rho_max = std::get<1>(bounds);

        /*
         * Verify that rho_U is within bounds. This property might be
         * violated for relative CFL numbers larger than 1.
         */
        const auto test_min = view.filter_vacuum_density(
            std::max(Number(0.), rho_U - relax * rho_max));
        const auto test_max = view.filter_vacuum_density(
            std::max(Number(0.), rho_min - relax * rho_U));
        if (!(test_min == Number(0.) && test_max == Number(0.))) {
#ifdef DEBUG_OUTPUT
          std::cout << std::fixed << std::setprecision(16);
          std::cout << "Bounds violation: low-order density (critical)!"
                    << "\n\t\trho min:         " << rho_min
                    << "\n\t\trho min (delta): "
                    << negative_part(rho_U - rho_min)
                    << "\n\t\trho:             " << rho_U
                    << "\n\t\trho max (delta): "
                    << positive_part(rho_U - rho_max)
                    << "\n\t\trho max:         " << rho_max << "\n"
                    << std::endl;
#endif
          success = false;
        }

        const Number denominator =
            ScalarNumber(1.) / (std::abs(rho_P) + eps * rho_max);

        t_r = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
            rho_max,
            rho_U + t_r * rho_P,
            /*
             * rho_P is positive.
             *
             * Note: Do not take an absolute value here. If we are out of
             * bounds we have to ensure that t_r is set to t_min.
             */
            (rho_max - rho_U) * denominator,
            t_r);

        t_r = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
            rho_U + t_r * rho_P,
            rho_min,
            /*
             * rho_P is negative.
             *
             * Note: Do not take an absolute value here. If we are out of
             * bounds we have to ensure that t_r is set to t_min.
             */
            (rho_U - rho_min) * denominator,
            t_r);

        /*
         * Ensure that t_min <= t <= t_max. This might not be the case if
         * rho_U is outside the interval [rho_min, rho_max]. Furthermore,
         * the quotient we take above is prone to numerical cancellation in
         * particular in the second pass of the limiter when rho_P might be
         * small.
         */
        t_r = std::min(t_r, t_max);
        t_r = std::max(t_r, t_min);

#ifdef EXPENSIVE_BOUNDS_CHECK
        /*
         * Verify that the new state is within bounds:
         */
        const auto rho_new = view.density(U + t_r * P);
        const auto test_new_min = view.filter_vacuum_density(
            std::max(Number(0.), rho_new - relax * rho_max));
        const auto test_new_max = view.filter_vacuum_density(
            std::max(Number(0.), rho_min - relax * rho_new));
        if (!(test_new_min == Number(0.) && test_new_max == Number(0.))) {
#ifdef DEBUG_OUTPUT
          std::cout << std::fixed << std::setprecision(16);
          std::cout << "Bounds violation: high-order density!"
                    << "\n\t\trho min:         " << rho_min
                    << "\n\t\trho min (delta): "
                    << negative_part(rho_new - rho_min)
                    << "\n\t\trho:             " << rho_new
                    << "\n\t\trho max (delta): "
                    << positive_part(rho_new - rho_max)
                    << "\n\t\trho max:         " << rho_max << "\n"
                    << std::endl;
#endif
          success = false;
        }
#endif
      }

      /*
       * Then limit the specific entropy:
       *
       * See [Guermond, Nazarov, Popov, Thomas],
       * Section 4.6 + Section 5.1
       * and @cite clayton2023robust Section 6:
       */

      Number t_l = t_min; // good state

      const auto &gamma = std::get<3>(bounds) /* = gamma_min*/;
      const Number gm1 = gamma - Number(1.);

      const auto b = Number(view.eos_interpolation_b());
      const auto pinf = Number(view.eos_interpolation_pinfty());
      const auto q = Number(view.eos_interpolation_q());

      {
        /*
         * Prepare a quadratic Newton method:
         *
         * Given initial limiter values t_l and t_r with psi(t_l) > 0 and
         * psi(t_r) < 0 we try to find t^\ast with psi(t^\ast) \approx 0.
         *
         * Here, psi is a 3-convex function obtained by scaling the specific
         * entropy s:
         *
         *   psi = \rho ^ {\gamma + 1} s
         *
         * (s in turn was defined as s =\varepsilon \rho ^{-\gamma}, where
         * \varepsilon = (\rho e - pinf * (1 - b rho)) is the shifted
         * internal energy.)
         */

        const auto &s_min = std::get<2>(bounds);

#ifdef DEBUG_OUTPUT_LIMITER
        std::cout << std::endl;
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "t_l: (start) " << t_l << std::endl;
        std::cout << "t_r: (start) " << t_r << std::endl;
#endif

        for (unsigned int n = 0; n < parameters.newton_max_iterations(); ++n) {

          const auto U_r = U + t_r * P;
          const auto rho_r = view.density(U_r);
          const auto rho_r_gamma = ryujin::pow(rho_r, gamma);
          const auto covolume_r = Number(1.) - b * rho_r;

          const auto rho_e_r = view.internal_energy(U_r);
          const auto shift_r = rho_e_r - rho_r * q - pinf * covolume_r;

          auto psi_r =
              relax_small * rho_r * shift_r -
              s_min * rho_r * rho_r_gamma * ryujin::pow(covolume_r, -gm1);

#ifndef EXPENSIVE_BOUNDS_CHECK
          /*
           * If psi_r > 0 the right state is fine, force returning t_r by
           * setting t_l = t_r:
           */
          t_l = dealii::compare_and_apply_mask<
              dealii::SIMDComparison::greater_than>(
              psi_r, Number(0.), t_r, t_l);

          /*
           * If we have set t_l = t_r everywhere then all states state U_r
           * with t_r obey the specific entropy inequality and we can
           * break.
           *
           * This is a very important optimization: Only for 1 in (25 to
           * 50) cases do we actually need to limit on the specific entropy
           * because one of the right states failed. So we can skip
           * constructing the left state U_l, which is expensive.
           *
           * This implies unfortunately that we might not accurately report
           * whether the low_order update U itself obeyed bounds because
           * U_r = U + t_r * P pushed us back into bounds. We thus skip
           * this shortcut if `EXPENSIVE_BOUNDS_CHECK` is set.
           */
          if (t_l == t_r) {
#ifdef DEBUG_OUTPUT_LIMITER
            std::cout << "shortcut: t_l == t_r" << std::endl;
            std::cout << "psi_l:       " << psi_l << std::endl;
            std::cout << "psi_r:       " << psi_r << std::endl;
            std::cout << "t_l: (  " << n << "  ) " << t_l << std::endl;
            std::cout << "t_r: (  " << n << "  ) " << t_r << std::endl;
#endif
            break;
          }
#endif

          const auto U_l = U + t_l * P;
          const auto rho_l = view.density(U_l);
          const auto rho_l_gamma = ryujin::pow(rho_l, gamma);
          const auto covolume_l = Number(1.) - b * rho_l;
          const auto rho_e_l = view.internal_energy(U_l);
          const auto shift_l = rho_e_l - rho_l * q - pinf * covolume_l;

          auto psi_l =
              relax_small * rho_l * shift_l -
              s_min * rho_l * rho_l_gamma * ryujin::pow(covolume_l, -gm1);

          /*
           * Verify that the left state is within bounds. This property might
           * be violated for relative CFL numbers larger than 1.
           */
          const auto lower_bound = (ScalarNumber(1.) - relax) * s_min * rho_l *
                                   rho_l_gamma * ryujin::pow(covolume_l, -gm1);
          if (n == 0 &&
              !(std::min(Number(0.), psi_l - lower_bound) == Number(0.))) {
#ifdef DEBUG_OUTPUT
            std::cout << std::fixed << std::setprecision(16);
            std::cout
                << "Bounds violation: low-order specific entropy (critical)!\n";
            std::cout << "\t\tPsi left: 0 <= " << psi_l << "\n" << std::endl;
#endif
            success = false;
          }

#ifdef EXPENSIVE_BOUNDS_CHECK
          /*
           * If psi_r > 0 the right state is fine, force returning t_r by
           * setting t_l = t_r:
           */
          t_l = dealii::compare_and_apply_mask<
              dealii::SIMDComparison::greater_than>(
              psi_r, Number(0.), t_r, t_l);
#endif

          /*
           * Break if the window between t_l and t_r is within the prescribed
           * tolerance:
           */
          const Number tolerance(parameters.newton_tolerance());
          if (std::max(Number(0.), t_r - t_l - tolerance) == Number(0.)) {
#ifdef DEBUG_OUTPUT_LIMITER
            std::cout << "break: t_l and t_r within tolerance" << std::endl;
            std::cout << "psi_l:       " << psi_l << std::endl;
            std::cout << "psi_r:       " << psi_r << std::endl;
            std::cout << "t_l: (  " << n << "  ) " << t_l << std::endl;
            std::cout << "t_r: (  " << n << "  ) " << t_r << std::endl;
#endif
            break;
          }

          /* We got unlucky and have to perform a Newton step: */

          const auto drho = view.density(P);
          const auto drho_e_l = view.internal_energy_derivative(U_l) * P;
          const auto drho_e_r = view.internal_energy_derivative(U_r) * P;

          const auto q_pinf_term_l =
              ScalarNumber(2.) * rho_l * q +
              pinf * (Number(1.) - ScalarNumber(2.) * b * rho_l);
          const auto q_pinf_term_r =
              ScalarNumber(2.) * rho_r * q +
              pinf * (Number(1.) - ScalarNumber(2.) * b * rho_r);

          const auto extra_term_l = s_min *
                                    ryujin::pow(rho_l / covolume_l, gamma) *
                                    (covolume_l + gamma - b * rho_l);
          const auto extra_term_r = s_min *
                                    ryujin::pow(rho_r / covolume_r, gamma) *
                                    (covolume_r + gamma - b * rho_r);

          const auto dpsi_l = rho_l * drho_e_l +
                              (rho_e_l - q_pinf_term_l - extra_term_l) * drho;
          const auto dpsi_r = rho_r * drho_e_r +
                              (rho_e_r - q_pinf_term_r - extra_term_r) * drho;

          quadratic_newton_step(
              t_l, t_r, psi_l, psi_r, dpsi_l, dpsi_r, Number(-1.));

#ifdef DEBUG_OUTPUT_LIMITER
          std::cout << "psi_l:       " << psi_l << std::endl;
          std::cout << "psi_r:       " << psi_r << std::endl;
          std::cout << "dpsi_l:      " << dpsi_l << std::endl;
          std::cout << "dpsi_r:      " << dpsi_r << std::endl;
          std::cout << "t_l: (  " << n << "  ) " << t_l << std::endl;
          std::cout << "t_r: (  " << n << "  ) " << t_r << std::endl;
#endif
        }

#ifdef EXPENSIVE_BOUNDS_CHECK
        /*
         * Verify that the new state is within bounds:
         */
        {
          const auto U_new = U + t_l * P;

          const auto rho_new = view.density(U_new);
          const auto covolume_new = Number(1.) - b * rho_new;

          const auto rho_new_gamma = ryujin::pow(rho_new, gamma);
          const auto rho_e_new = view.internal_energy(U_new);

          const auto shift_new = rho_e_new - rho_new * q - pinf * covolume_new;

          const auto psi_new =
              relax_small * rho_new * shift_new -
              s_min * rho_new * rho_new_gamma * ryujin::pow(covolume_new, -gm1);

          const auto lower_bound = (ScalarNumber(1.) - relax) * s_min *
                                   rho_new * rho_new_gamma *
                                   ryujin::pow(covolume_new, -gm1);

          const bool e_valid = std::min(Number(0.), shift_new) == Number(0.);
          const bool psi_valid =
              std::min(Number(0.), psi_new - lower_bound) == Number(0.);

          if (!e_valid || !psi_valid) {
#ifdef DEBUG_OUTPUT
            std::cout << std::fixed << std::setprecision(16);
            std::cout << "Bounds violation: high-order specific entropy!\n";
            std::cout << "\t\trho e: 0 <= " << rho_e_new << "\n";
            std::cout << "\t\tPsi:   0 <= " << psi_new << "\n" << std::endl;
#endif
            success = false;
          }
        }
#endif
      }

      return {t_l, success};
    }

  } // namespace EulerAEOS
} // namespace ryujin
