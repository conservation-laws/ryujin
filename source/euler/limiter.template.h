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
                              const ScalarNumber newton_tolerance,
                              const unsigned int newton_max_iter,
                              const Number t_min /* = Number(0.) */,
                              const Number t_max /* = Number(1.) */)
  {
    bool success = true;
    Number t_r = t_max;

    constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
    constexpr ScalarNumber relax = ScalarNumber(1. + 10. * eps);
    constexpr ScalarNumber relaxbig = ScalarNumber(1. + 10000. * eps);

    /*
     * First limit the density rho.
     *
     * See [Guermond, Nazarov, Popov, Thomas] (4.8):
     */

    {
      const auto &U_rho = hyperbolic_system.density(U);
      const auto &P_rho = hyperbolic_system.density(P);

      const auto &rho_min = std::get<0>(bounds);
      const auto &rho_max = std::get<1>(bounds);

      /*
       * Verify that U_rho is within bounds. This property might be
       * violated for relative CFL numbers larger than 1.
       */
      if (!((std::max(Number(0.), U_rho - relaxbig * rho_max) == Number(0.)) &&
            (std::max(Number(0.), rho_min - relaxbig * U_rho) == Number(0.)))) {
#ifdef DEBUG_OUTPUT
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Bounds violation: low-order density (critical)!\n";
        std::cout << "\t\trho min: " << rho_min << "\n";
        std::cout << "\t\trho:     " << U_rho << "\n";
        std::cout << "\t\trho max: " << rho_max << "\n" << std::endl;
#endif
        success = false;
      }

      const Number denominator =
          ScalarNumber(1.) / (std::abs(P_rho) + eps * rho_max);
      t_r = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
          rho_max,
          U_rho + t_r * P_rho,
          (std::abs(rho_max - U_rho) + eps * rho_min) * denominator,
          t_r);

      t_r = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
          U_rho + t_r * P_rho,
          rho_min,
          (std::abs(rho_min - U_rho) + eps * rho_min) * denominator,
          t_r);

      /*
       * It is always t_min <= t <= t_max, but just to be sure, box
       * back into bounds:
       */
      t_r = std::min(t_r, t_max);
      t_r = std::max(t_r, t_min);

#ifdef CHECK_BOUNDS
      /*
       * Verify that the new state is within bounds:
       */
      const auto n_rho = hyperbolic_system.density(U + t_r * P);
      if (!((std::max(Number(0.), n_rho - relaxbig * rho_max) == Number(0.)) &&
            (std::max(Number(0.), rho_min - relaxbig * n_rho) == Number(0.)))) {
#ifdef DEBUG_OUTPUT
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Bounds violation: high-order density!\n";
        std::cout << "\t\trho min: " << rho_min << "\n";
        std::cout << "\t\trho:     " << n_rho << "\n";
        std::cout << "\t\trho max: " << rho_max << "\n" << std::endl;
#endif
        success = false;
      }
#endif
    }

    /*
     * Then limit the specific entropy:
     *
     * See [Guermond, Nazarov, Popov, Thomas], Section 4.6 + Section 5.1:
     */

    Number t_l = t_min; // good state

    const ScalarNumber gamma = hyperbolic_system.gamma();
    const ScalarNumber gp1 = gamma + ScalarNumber(1.);

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
       * \varepsilon = (\rho e) is the internal energy.)
       */

      const auto &s_min = std::get<2>(bounds);

      for (unsigned int n = 0; n < newton_max_iter; ++n) {

        const auto U_r = U + t_r * P;
        const auto rho_r = hyperbolic_system.density(U_r);
        const auto rho_r_gamma = ryujin::pow(rho_r, gamma);
        const auto rho_e_r = hyperbolic_system.internal_energy(U_r);

        auto psi_r = relax * rho_r * rho_e_r - s_min * rho_r * rho_r_gamma;

        /* If psi_r > 0 the right state is fine, force returning t_r by
         * setting t_l = t_r: */
        t_l = dealii::compare_and_apply_mask<
            dealii::SIMDComparison::greater_than>(psi_r, Number(0.), t_r, t_l);

        /* If we have set t_l = t_r everywhere we can break: */
        if (t_l == t_r)
          break;

        const auto U_l = U + t_l * P;
        const auto rho_l = hyperbolic_system.density(U_l);
        const auto rho_l_gamma = ryujin::pow(rho_l, gamma);
        const auto rho_e_l = hyperbolic_system.internal_energy(U_l);

        auto psi_l = relax * rho_l * rho_e_l - s_min * rho_l * rho_l_gamma;

        /*
         * Verify that the left state is within bounds. This property might
         * be violated for relative CFL numbers larger than 1.
         */
        if (n == 0 &&
            !(std::min(Number(0.), psi_l + Number(100. * eps)) == Number(0.))) {
#ifdef DEBUG_OUTPUT
          std::cout << std::fixed << std::setprecision(16);
          std::cout
              << "Bounds violation: low-order specific entropy (critical)!\n";
          std::cout << "\t\tPsi left: 0 <= " << psi_l << "\n" << std::endl;
#endif
          success = false;
        }

        /* Break if all psi_l values are within a prescribed tolerance: */
        if (std::max(
                Number(0.),
                dealii::compare_and_apply_mask<
                    dealii::SIMDComparison::greater_than>(
                    psi_r, Number(0.), Number(0.), psi_l - newton_tolerance)) ==
            Number(0.))
          break;

        /* We got unlucky and have to perform a Newton step: */

        const auto drho = hyperbolic_system.density(P);
        const auto drho_e_l =
            hyperbolic_system.internal_energy_derivative(U_l) * P;
        const auto drho_e_r =
            hyperbolic_system.internal_energy_derivative(U_r) * P;
        const auto dpsi_l =
            rho_l * drho_e_l + (rho_e_l - gp1 * s_min * rho_l_gamma) * drho;
        const auto dpsi_r =
            rho_r * drho_e_r + (rho_e_r - gp1 * s_min * rho_r_gamma) * drho;

        quadratic_newton_step(
            t_l, t_r, psi_l, psi_r, dpsi_l, dpsi_r, Number(-1.));

        /* Let's error on the safe side: */
        t_l -= ScalarNumber(0.2) * newton_tolerance;
        t_r += ScalarNumber(0.2) * newton_tolerance;
      }

#ifdef CHECK_BOUNDS
      /*
       * Verify that the new state is within bounds:
       */
      {
        const auto U_new = U + t_l * P;
        const auto rho_new = hyperbolic_system.density(U_new);
        const auto e_new = hyperbolic_system.internal_energy(U_new);
        const auto psi =
            relax * relax * rho_new * e_new - s_min * ryujin::pow(rho_new, gp1);

        const bool e_valid = std::min(Number(0.), e_new) == Number(0.);
        const bool psi_valid =
            std::min(Number(0.), psi + ScalarNumber(100.) * eps) == Number(0.);
        if (!e_valid || !psi_valid) {
#ifdef DEBUG_OUTPUT
          std::cout << std::fixed << std::setprecision(16);
          std::cout << "Bounds violation: high-order specific entropy!\n";
          std::cout << "\t\tInt: 0 <= " << e_new << "\n";
          std::cout << "\t\tPsi: 0 <= " << psi << "\n" << std::endl;
#endif
          success = false;
        }
      }
#endif
    }

    return {t_l, success};
  }

} // namespace ryujin
