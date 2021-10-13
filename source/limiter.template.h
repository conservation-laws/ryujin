//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include "limiter.h"

namespace ryujin
{
  template <int dim, typename Number>
  template <typename Limiter<dim, Number>::Limiters limiter, typename BOUNDS>
#ifdef OBSESSIVE_INLINING
  DEAL_II_ALWAYS_INLINE inline
#endif
      std::tuple<Number, bool>
      Limiter<dim, Number>::limit(const ProblemDescription &problem_description,
                                  const BOUNDS &bounds,
                                  const state_type &U,
                                  const state_type &P,
                                  const Number t_min /* = Number(0.) */,
                                  const Number t_max /* = Number(1.) */)
  {
    bool success = true;
    Number t_r = t_max;

    if constexpr (limiter == Limiters::none)
      return {t_r, success};

    constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
    constexpr ScalarNumber relax = ScalarNumber(1.) + 10. * eps;
    constexpr ScalarNumber relaxbig = ScalarNumber(1.) + 3000. * eps;

    /*
     * First limit the density rho.
     *
     * See [Guermond, Nazarov, Popov, Thomas] (4.8):
     */

    {
      const auto &U_rho = problem_description.density(U);
      const auto &P_rho = problem_description.density(P);

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
        std::cout << "(crit) min: " << rho_min << std::endl;
        std::cout << "(crit) rho: " << U_rho << std::endl;
        std::cout << "(crit) max: " << rho_max << std::endl << std::endl;
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
      const auto n_rho = problem_description.density(U + t_r * P);
      if (!((std::max(Number(0.), n_rho - relaxbig * rho_max) == Number(0.)) &&
            (std::max(Number(0.), rho_min - relaxbig * n_rho) == Number(0.)))) {
#ifdef DEBUG_OUTPUT
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Density bounds violated:" << std::endl;
        std::cout << "min: " << rho_min << std::endl;
        std::cout << "rho: " << n_rho << std::endl;
        std::cout << "max: " << rho_max << std::endl << std::endl;
#endif
        success = false;
      }
#endif
    }

    if constexpr (limiter == Limiters::rho)
      return {t_r, success};

    /*
     * Then limit the specific entropy:
     *
     * See [Guermond, Nazarov, Popov, Thomas], Section 4.6 + Section 5.1:
     */

    Number t_l = t_min; // good state

    const ScalarNumber gamma = problem_description.gamma();
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
        const auto rho_r = problem_description.density(U_r);
        const auto rho_r_gamma = ryujin::pow(rho_r, gamma);
        const auto rho_e_r = problem_description.internal_energy(U_r);

        auto psi_r = relax * rho_r * rho_e_r - s_min * rho_r * rho_r_gamma;

        /* If psi_r > 0 the right state is fine, force returning t_r by
         * setting t_l = t_r: */
        t_l = dealii::compare_and_apply_mask<
            dealii::SIMDComparison::greater_than>(psi_r, Number(0.), t_r, t_l);

        /* If we have set t_l = t_r everywhere we can break: */
        if (t_l == t_r)
          break;

        const auto U_l = U + t_l * P;
        const auto rho_l = problem_description.density(U_l);
        const auto rho_l_gamma = ryujin::pow(rho_l, gamma);
        const auto rho_e_l = problem_description.internal_energy(U_l);

        auto psi_l = relax * rho_l * rho_e_l - s_min * rho_l * rho_l_gamma;

        /*
         * Verify that the left state is within bounds. This property might
         * be violated for relative CFL numbers larger than 1.
         */
        if (n == 0 &&
            !(std::min(Number(0.), psi_l + 100. * eps) == Number(0.))) {
#ifdef DEBUG_OUTPUT
          std::cout << std::fixed << std::setprecision(16);
          std::cout << "(crit) Psi left: " << psi_l << std::endl;
#endif
          success = false;
        }

        /* Break if all psi_l values are within a prescribed tolerance: */
        if (std::max(Number(0.),
                     dealii::compare_and_apply_mask<
                         dealii::SIMDComparison::greater_than>(
                         psi_r,
                         Number(0.),
                         Number(0.),
                         psi_l - newton_eps<Number>)) == Number(0.))
          break;

        /* We got unlucky and have to perform a Newton step: */

        const auto drho = problem_description.density(P);
        const auto drho_e_l =
            problem_description.internal_energy_derivative(U_l) * P;
        const auto drho_e_r =
            problem_description.internal_energy_derivative(U_r) * P;
        const auto dpsi_l =
            rho_l * drho_e_l + (rho_e_l - gp1 * s_min * rho_l_gamma) * drho;
        const auto dpsi_r =
            rho_r * drho_e_r + (rho_e_r - gp1 * s_min * rho_r_gamma) * drho;

        quadratic_newton_step(
            t_l, t_r, psi_l, psi_r, dpsi_l, dpsi_r, Number(-1.));

        /* Let's error on the safe side: */
        t_l -= ScalarNumber(0.2) * newton_eps<Number>;
        t_r += ScalarNumber(0.2) * newton_eps<Number>;
      }

#ifdef CHECK_BOUNDS
      /*
       * Verify that the new state is within bounds:
       */
      {
        const auto U_new = U + t_l * P;
        const auto rho_new = problem_description.density(U_new);
        const auto e_new = problem_description.internal_energy(U_new);
        const auto psi =
            relax * relax * rho_new * e_new - s_min * ryujin::pow(rho_new, gp1);

        if (!((std::min(Number(0.), e_new) == Number(0.)) &&
              (std::min(Number(0.), psi + 100. * eps) == Number(0.))

                  )) {
#ifdef DEBUG_OUTPUT
          std::cout << std::fixed << std::setprecision(16);
          std::cout << "Specific entropy minimum principle violated:"
                    << std::endl;
          std::cout << "int: !!! 0 <= " << e_new << std::endl;
          std::cout << "Psi: !!! 0 <= " << psi << std::endl << std::endl;
#endif
          success = false;
        }
      }
#endif
    }

    if constexpr (limiter == Limiters::specific_entropy)
      return {t_l, success};

    /*
     * Limit based on an entropy inequality:
     *
     * We use a Harten-type entropy (with alpha = 1) of the form:
     *
     *   salpha  = (rho^2 e) ^ (1 / (gamma + 1))
     *
     * Then, given an average
     *
     *   a  =  0.5 * (salpha_i + salpha_j)
     *
     * and a flux
     *
     *   b  =  0.5 * (u_i * salpha_i - u_j * salpha_j)
     *
     * The task is to find the maximal t within the bounds [t_l, t_r]
     * (which happens to be equal to our old bounds [t_min, t_l]) such that
     *
     *   psi(t) = (a + bt)^(gamma + 1) - \rho^2 e (t) <= 0.
     *
     * Note that we have to take care of one small corner case: The above
     * line search (with quadratic Newton) will fail if (a + b t) < 0. In
     * this case, however we have that that t_0 < t_r for t_0 = - a / b,
     * this implies that t_r is already a good state. We thus simply modify
     * psi(t) to
     *
     *  psi(t) = ([a + bt]_pos)^(gamma + 1) - rho^2 e (t)
     *
     * in this case because this immediately accepts the right state.
     */

    static_assert(limiter != Limiters::entropy_inequality, "codepath untested");

    t_r = t_l;   // bad state
    t_l = t_min; // good state

    if constexpr (limiter == Limiters::entropy_inequality) {
      const auto a = std::get<3>(bounds); /* 0.5*(salpha_i+salpha_j) */
      const auto b = std::get<4>(bounds); /* 0.5*(u_i*salpha_i-u_j*salpha_j) */

      /* Extract the sign of psi''' depending on b: */
      const auto sign =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
              b, Number(0.), Number(-1.), Number(1.));

      /*
       * Same quadratic Newton method as above, but for a different
       * 3-concave/3-convex psi:
       *
       *  psi(t) = ([a + bt]_pos)^(gamma + 1) - rho^2 e (t)
       */

      for (unsigned int n = 0; n < newton_max_iter; ++n) {

        const auto U_r = U + t_r * P;
        const auto rho_r = problem_description.density(U_r);
        const auto rho_e_r = problem_description.internal_energy(U_r);
        const auto average_r = positive_part(a + b * t_r);
        const auto average_gamma_r = std::pow(average_r, gamma);

        auto psi_r = average_r * average_gamma_r - rho_r * rho_e_r * relax;

        /* If psi_r <= 0 the right state is fine, force returning t_r by
         * setting t_l = t_r: */
        t_l = dealii::compare_and_apply_mask<
            dealii::SIMDComparison::less_than_or_equal>(
            psi_r, Number(0.), t_r, t_l);

        /* If we have set t_l = t_r we can break: */
        if (t_l == t_r)
          break;

        const auto U_l = U + t_l * P;
        const auto rho_l = problem_description.density(U_l);
        const auto rho_e_l = problem_description.internal_energy(U_l);
        const auto average_l = positive_part(a + b * t_l);
        const auto average_gamma_l = std::pow(average_l, gamma);

        auto psi_l = average_l * average_gamma_l - rho_l * rho_e_l * relax;

        /* Break if all psi_l values are within a prescribed tolerance: */
        if (std::min(Number(0.),
                     dealii::compare_and_apply_mask<
                         dealii::SIMDComparison::less_than_or_equal>(
                         psi_r,
                         Number(0.),
                         Number(0.),
                         psi_l + newton_eps<Number>)) == Number(0.))
          break;

        /* We got unlucky and have to perform a Newton step: */

        const auto drho = problem_description.density(P);
        const auto drho_e_l =
            problem_description.internal_energy_derivative(U_l) * P;
        const auto drho_e_r =
            problem_description.internal_energy_derivative(U_r) * P;

        const auto dpsi_l =
            gp1 * average_gamma_l * b - rho_e_l * drho - rho_l * drho_e_l;

        const auto dpsi_r =
            gp1 * average_gamma_r * b - rho_e_r * drho - rho_r * drho_e_r;

#ifdef CHECK_BOUNDS
        AssertThrowSIMD(
            psi_l,
            [](auto val) { return val < 1000. * eps; },
            dealii::ExcMessage("Entropy inequality violated."));
#endif

        quadratic_newton_step(t_l, t_r, psi_l, psi_r, dpsi_l, dpsi_r, sign);

        /* Let's error on the safe side: */
        t_l -= ScalarNumber(0.2) * newton_eps<Number>;
        t_r += ScalarNumber(0.2) * newton_eps<Number>;
      }

#ifdef CHECK_BOUNDS
      const auto U_new = U + t_l * P;
      const auto rho_rho_e = problem_description.density(U_new) *
                             problem_description.internal_energy(U_new);
      const auto avg = ryujin::pow(positive_part(a + b * t_l), gp1);

      AssertThrowSIMD(
          avg - rho_rho_e * relax,
          [](auto val) { return val < 1000. * eps; },
          dealii::ExcMessage("Entropy inequality violated."));
#endif
    }

    return {t_l, success};
  }

} /* namespace ryujin */
