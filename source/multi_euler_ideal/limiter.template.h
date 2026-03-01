//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2025 by the ryujin authors
// Copyright (C) 2025 by Triad National Security, LLC
//

#pragma once

#include "limiter.h"
// #define DEBUG_OUTPUT_LIMITER

namespace ryujin
{
  namespace MultiSpeciesEuler
  {
    template <int dim, typename Number>
    std::tuple<Number, bool>
    Limiter<dim, Number>::limit(const Bounds &bounds,
                                const state_type &U,
                                const state_type &P,
                                const Number t_min /* = Number(0.) */,
                                const Number t_max /* = Number(1.) */) const
    {
      const auto view = hyperbolic_system.view<dim, Number>();

      bool success = true;
      Number t_r = t_max;

      constexpr ScalarNumber min = std::numeric_limits<ScalarNumber>::min();
      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
      const auto small = view.vacuum_state_relaxation_small();
      const auto large = view.vacuum_state_relaxation_large();
      const ScalarNumber relax_small = ScalarNumber(1. + small * eps);
      const ScalarNumber relax = ScalarNumber(1. + large * eps);

      /*
       * Limit the partial densities for each species.
       *
       * See [Guermond, Nazarov, Popov, Thomas] (4.8):
       */

      for (unsigned int k = 0; k < n_species; ++k) {
        const auto rho_U = view.partial_density(U, k);
        const auto rho_P = view.partial_density(P, k);

        const auto &rho_min = bounds[2 * k];
        const auto &rho_max = bounds[2 * k + 1];

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
          std::cout << "Bounds violation: low-order [species " << k
                    << "] density (critical)!"
                    << "\n\t\trho min:         " << rho_min
                    << "\n\t\trho min (delta): " << negative_part(rho_U - rho_min)
                    << "\n\t\trho:             " << rho_U
                    << "\n\t\trho max (delta): " << positive_part(rho_U - rho_max)
                    << "\n\t\trho max:         " << rho_max << "\n"
                    << std::endl;
#endif
          success = false;
        }

        const Number denominator =
            ScalarNumber(1.) / (std::abs(rho_P) + eps * rho_max + min);

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

#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
        /*
         * Verify that the new state is within bounds:
         */
        const auto rho_new = view.partial_density(U + t_r * P, k);
        const auto test_new_min = view.filter_vacuum_density(
            std::max(Number(0.), rho_new - relax * rho_max));
        const auto test_new_max = view.filter_vacuum_density(
            std::max(Number(0.), rho_min - relax * rho_new));
        if (!(test_new_min == Number(0.) && test_new_max == Number(0.))) {
#ifdef DEBUG_OUTPUT
          std::cout << std::fixed << std::setprecision(16);
          std::cout << "Bounds violation: high-order [species " << k
                    << "] density!"
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
      } /* end loop over species */

      /*
       * Then limit the internal energy. Unfortunately, limiting on the
       * "mixture" specific entropy does not guarantee positivity of the mixture
       * internal energy, so have to do this.
       */

      {
        Number t_l = t_min; // good state
        Number t_q = t_r;   // potentially bad state

        const auto &rho_e_min = bounds[2 * n_species];

        const auto U_l = U + t_l * P;
        const auto rho_e_l = view.internal_energy(U_l);
        auto psi_l = relax_small * rho_e_l - rho_e_min;

        /*
         * Verify that the left state is within bounds. This property might
         * be violated for relative CFL numbers larger than 1. The small number
         * of 1e-10 is sort of arbitrary, but dealing with round off is annoying
         * anyway.
         */
        const auto lower_bound = (ScalarNumber(1.) - relax) * rho_e_min;
        if (!(std::min(Number(0.), psi_l - lower_bound + Number(1.e-10)) ==
              Number(0.))) {
#ifdef DEBUG_OUTPUT
          std::cout << std::fixed << std::setprecision(16);
          std::cout
              << "Bounds violation: low-order internal energy (critical)!\n";
          std::cout << "\t\tPsi left: 0 <= " << psi_l << "\n" << std::endl;
#endif
          success = false;
        }

        const auto U_r = U + t_q * P;
        const auto rho_e_r = view.internal_energy(U_r);

        auto psi_r = relax_small * rho_e_r - rho_e_min;

        // Compute simple "linear" limiter
        const auto new_t_q = dealii::compare_and_apply_mask<
            dealii::SIMDComparison::greater_than>(std::abs(psi_l - psi_r),
                                                  Number(0.),
                                                  (-psi_r * t_l + psi_l * t_r) /
                                                      (psi_l - psi_r),
                                                  t_r);

        t_q = dealii::compare_and_apply_mask<
            dealii::SIMDComparison::greater_than>(
            psi_r, Number(0.), t_r, new_t_q);

        t_r = t_q;
        t_r = std::min(t_r, t_max);
        t_r = std::max(t_r, t_min);

#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
        /*
         * Verify that the new state is within bounds:
         */
        {
          const auto U_new = U + t_r * P;
          const auto rho_e_new = view.internal_energy(U_new);
          const bool rho_e_valid =
              std::min(Number(0.), rho_e_new) == Number(0.);

          if (!rho_e_valid) {
#ifdef DEBUG_OUTPUT
            std::cout << std::fixed << std::setprecision(16);
            std::cout << "Bounds violation: high-order internal energy!\n";
            std::cout << "\t\trho e: 0 <= " << rho_e_new << "\n";
#endif
            success = false;
          }
        }
#endif


      } // end limiting on internal energy

      /*
       * Then limit the concave entropy functional. This approach is not
       * documented anywhere. It is a simple approach that works only under the
       * assumptions that:
       * 1. The entropy (rho * sbar) is concave.
       * 2. The low-order method satisfies the minimum principle (which we can
       * prove)
       * We do this because we can't take derivatives of the true psi
       * functional. It's a mess.
       * The idea is as follows: Given the concave functional:
       * psi(u) = rho s - rho s_min
       * we can show that: psi(t) = rho(U + t P) s(U + t P) - rho(U + t P) s_min
       * is also concave. Then, we we find a root of psi(t) = 0, with a simple
       * linear search. That is:
       * given g(t) = psi(t_L) + (t - t_L) (psi(t_R) - psi(t_L)) / (t_R - t_L)
       * we get g(t) = 0 from t->(-psi_r * t_l + psi_l * t_r) / (psi_l - psi_r)
       */

      Number t_l = t_min; // good state

      {
        const auto &s_min = bounds[2 * n_species + 1];

#ifdef DEBUG_OUTPUT_LIMITER
        std::cout << std::endl;
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "t_l: (start) " << t_l << std::endl;
        std::cout << "t_r: (start) " << t_r << std::endl;
#endif

        const auto U_r = U + t_r * P;
        const auto rho_r = view.density(U_r);
        const auto s_r = view.specific_entropy(U_r);

        auto psi_r = relax_small * s_r * rho_r - s_min * rho_r;

#ifndef DEBUG_EXPENSIVE_BOUNDS_CHECK
        /*
         * If psi_r > 0 the right state is fine, force returning t_r by
         * setting t_l = t_r:
         */
        t_l = dealii::compare_and_apply_mask<
            dealii::SIMDComparison::greater_than>(psi_r, Number(0.), t_r, t_l);

        /*
         * If we have set t_l = t_r everywhere then all states state U_r
         * with t_r obey the specific entropy inequality and we can
         * return.
         *
         * This is a very important optimization: Only for 1 in (25 to
         * 50) cases do we actually need to limit on the specific entropy
         * because one of the right states failed. So we can skip
         * constructing the left state U_l, which is expensive.
         *
         * This implies unfortunately that we might not accurately report
         * whether the low_order update U itself obeyed bounds because
         * U_r = U + t_r * P pushed us back into bounds. We thus skip
         * this shortcut if `DEBUG_EXPENSIVE_BOUNDS_CHECK` is set.
         */
        if (t_l == t_r) {
#ifdef DEBUG_OUTPUT_LIMITER
          std::cout << "shortcut: t_l == t_r" << std::endl;
          std::cout << "psi_r:       " << psi_r << std::endl;
          std::cout << "t_l:         " << t_l << std::endl;
          std::cout << "t_r:         " << t_r << std::endl;
#endif
          return {t_l, success};
        }
#endif

        const auto U_l = U + t_l * P;
        const auto rho_l = view.density(U_l);
        const auto s_l = view.specific_entropy(U_l);

        auto psi_l = relax_small * s_l * rho_l - s_min * rho_l;

        /*
         * Verify that the left state is within bounds. This property might
         * be violated for relative CFL numbers larger than 1. The small number
         * of 1e-10 is sort of arbitrary, but dealing with round off is annoying
         * anyway.
         */
        const auto lower_bound = (ScalarNumber(1.) - relax) * s_min * rho_l;
        if (!(std::min(Number(0.), psi_l - lower_bound + Number(1.e-10)) ==
              Number(0.))) {
#ifdef DEBUG_OUTPUT
          std::cout << std::fixed << std::setprecision(16);
          std::cout
              << "Bounds violation: low-order specific entropy (critical)!\n";
          std::cout << "\t\tPsi left: 0 <= " << psi_l << "\n" << std::endl;
#endif
          success = false;
        }

#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
        /*
         * If psi_r > 0 the right state is fine, force returning t_r by
         * setting t_l = t_r:
         */
        t_l = dealii::compare_and_apply_mask<
            dealii::SIMDComparison::greater_than>(psi_r, Number(0.), t_r, t_l);
#endif

        /*
         * Return if the window between t_l and t_r is within the prescribed
         * tolerance:
         */
        const Number tolerance(parameters.newton_tolerance());
        if (std::max(Number(0.), t_r - t_l - tolerance) == Number(0.)) {
#ifdef DEBUG_OUTPUT_LIMITER
          std::cout << "break: t_l and t_r within tolerance" << std::endl;
          std::cout << "psi_l:       " << psi_l << std::endl;
          std::cout << "psi_r:       " << psi_r << std::endl;
          std::cout << "t_l:         " << t_l << std::endl;
          std::cout << "t_r:         " << t_r << std::endl;
#endif
          return {t_l, success};
        }

        /* We got unlucky and have to set t_l with linear formula: */

        t_l = dealii::compare_and_apply_mask<
            dealii::SIMDComparison::greater_than>(std::abs(psi_l - psi_r),
                                                  Number(0.),
                                                  (-psi_r * t_l + psi_l * t_r) /
                                                      (psi_l - psi_r),
                                                  t_r);

#ifdef DEBUG_OUTPUT_LIMITER
        std::cout << "s_min:       " << s_min << std::endl;
        std::cout << "psi_l:       " << psi_l << std::endl;
        std::cout << "psi_r:       " << psi_r << std::endl;
        std::cout << "t_l (new):   " << t_l << std::endl;
        std::cout << "t_r:         " << t_r << std::endl;
#endif


#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
        /*
         * Verify that the new state is within bounds:
         */
        {
          const auto U_new = U + t_l * P;
          const auto rho_new = view.density(U_new);
          const auto s_new = view.specific_entropy(U_new);
          const auto rho_e_new = view.internal_energy(U_new);

          auto psi_new = relax_small * s_new * rho_new - s_min * rho_new;

          const auto lower_bound = (ScalarNumber(1.) - relax) * s_min * rho_new;

          const bool e_valid = std::min(Number(0.), rho_e_new) == Number(0.);
          const bool psi_valid =
              std::min(Number(0.), psi_new - lower_bound) == Number(0.);

          if (!e_valid || !psi_valid) {
#ifdef DEBUG_OUTPUT
            std::cout << std::fixed << std::setprecision(16);
            std::cout << "Bounds violation: high-order specific entropy!\n";
            std::cout << "\t\trho e: 0 <= " << rho_e_new << "\n";
            std::cout << "\t\tPsi:   0 <= " << psi_new << "\n" << std::endl;
            std::cout << "\t\tLow:        " << lower_bound << "\n" << std::endl;
            std::cout << "\t\tDiff:       " << psi_new - lower_bound << "\n"
                      << std::endl;
#endif
            success = false;
          }
        }
#endif
      }

      return {t_l, success};
    }

  } // namespace MultiSpeciesEuler
} // namespace ryujin
