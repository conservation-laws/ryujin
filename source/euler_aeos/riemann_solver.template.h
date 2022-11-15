//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "riemann_solver.h"

#include <newton.h>
#include <simd.h>

namespace ryujin
{
  namespace EulerAEOS
  {
    /*
     * This is a modification of the RiemannSolver Class in ryujin/euler. We
     * refer the reader to:
     *
     * Invariant Domain-Preserving approximations for the Euler equations
     * with tabulated equation of state, Bennett Clayton, Jean-Luc Guermond,
     * and Bojan * Popov, SIAM Journal on Scientific Computing 2022 44:1,
     * A444-A470
     */

    /* Compute \alpha_Z / c(\gamma_Z) function */

    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number RiemannSolver<dim, Number>::alpha(
        const Number &rho, const Number &gamma, const Number &a) const
    {
      using ScalarNumber = typename get_value_type<Number>::type;

      const ScalarNumber b_interp = hyperbolic_system.b_interp();

      const Number numerator =
          ScalarNumber(2.) * a * (Number(1.) - b_interp * rho);

      const Number denominator = gamma - Number(1.);

      return numerator / denominator;
    }

    /* Compute c(\gamma_Z) function */

    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::c(const Number gamma_Z) const
    {
      using ScalarNumber = typename get_value_type<Number>::type;

      Number radicand = (ScalarNumber(3.) * gamma_Z + Number(11.)) /
                        (ScalarNumber(6.) * (gamma_Z + Number(1.)));
      const Number false_value = std::sqrt(radicand);

      Number c_of_gamma = dealii::compare_and_apply_mask<
          dealii::SIMDComparison::less_than_or_equal>(
          gamma_Z, Number(5. / 3.), Number(1.), false_value);

      const Number true_value = Number(0.5 * std::sqrt(2.));

      c_of_gamma = dealii::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          gamma_Z, Number(3.), true_value, c_of_gamma);

      return c_of_gamma;
    }

    /* For p_star in the Expansion/Shock case. Pessimistic approximation */
    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::p_star_RS_aeos(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      using ScalarNumber = typename get_value_type<Number>::type;

      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;
      const auto alpha_i = alpha(rho_i, gamma_i, a_i);
      const auto alpha_j = alpha(rho_j, gamma_j, a_j);

      //
      // First get p_min, p_max.
      //
      // Then, we get gamma_min/max, and alpha_min/max. Note that the
      // *_min/max values are associated with p_min/max and are not
      // necessarily the minimum/maximum of *_i vs *_j.
      //

      const Number p_min = std::min(p_i, p_j);
      const Number p_max = std::max(p_i, p_j);

      const Number gamma_min =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, gamma_i, gamma_j);

      const Number gamma_max = dealii::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_i, p_j, gamma_i, gamma_j);

      const Number alpha_min =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, alpha_i, alpha_j);

      const Number alpha_max = dealii::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_i, p_j, alpha_i, alpha_j);

      const Number c_gamma_min = c(gamma_min);

      const Number exp_min =
          ScalarNumber(2.) * gamma_min / (gamma_min - Number(1.));
      const Number exp_max =
          (gamma_max - Number(1.)) / (ScalarNumber(2.) * gamma_max);

      /* Then we can compute p_star_RS */
      const Number numerator =
          alpha_max * (Number(1.) - ryujin::vec_pow(p_min / p_max, exp_max)) -
          (u_j - u_i);

      const Number denominator = c_gamma_min * alpha_min;

      const Number base = numerator / denominator + Number(1.);

      return p_min * ryujin::vec_pow(base, exp_min);
    }

    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::p_star_SS_aeos(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      using ScalarNumber = typename get_value_type<Number>::type;

      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;

      const Number gamma_m = std::min(gamma_i, gamma_j);

      /* Compute alpha_hat_left and alpha_hat_right  */
      const Number alpha_hat_left = c(gamma_i) * alpha(rho_i, gamma_i, a_i);
      const Number alpha_hat_right = c(gamma_j) * alpha(rho_j, gamma_j, a_j);

      const Number exp = (gamma_m - Number(1.)) / (ScalarNumber(2.) * gamma_m);
      const Number exp_inv = Number(1.) / exp;

      /* Then we can compute p_star_SS */
      const Number numerator = alpha_hat_left + alpha_hat_right - (u_j - u_i);

      const Number denominator = alpha_hat_left * ryujin::vec_pow(p_i, -exp) +
                                 alpha_hat_right * ryujin::vec_pow(p_j, -exp);

      const Number base = numerator / denominator;

      return ryujin::vec_pow(base, exp_inv);
    }

    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::f(const primitive_type &primitive_state,
                                  const Number p_star) const
    {
      using ScalarNumber = typename get_value_type<Number>::type;
      const ScalarNumber b_interp = hyperbolic_system.b_interp();

      const auto &[rho, u, p, gamma, a] = primitive_state;

      const Number g_minus_one = gamma - Number(1.);
      const Number g_plus_one = gamma + Number(1.);

      const Number one_minus_b_rho = Number(1.) - b_interp * rho;

      const Number Az =
          ScalarNumber(2.) * one_minus_b_rho / (rho * (g_plus_one));
      const Number Bz = g_minus_one / g_plus_one * p;
      const Number radicand = Az / (p_star + Bz);

      const Number true_value = (p_star - p) * std::sqrt(radicand);

      const auto exponent = ScalarNumber(0.5) * g_minus_one / gamma;
      const Number factor = ryujin::vec_pow(p_star / p, exponent) - Number(1.);
      const auto false_value =
          ScalarNumber(2.) * a * one_minus_b_rho * factor / g_minus_one;

      return dealii::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_star, p, true_value, false_value);
    }

    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::phi_of_p(const primitive_type &riemann_data_i,
                                         const primitive_type &riemann_data_j,
                                         const Number p_in) const
    {
      const Number &u_i = riemann_data_i[1];
      const Number &u_j = riemann_data_j[1];

      return f(riemann_data_i, p_in) + f(riemann_data_j, p_in) + u_j - u_i;
    }

    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::lambda1_minus(
        const primitive_type &riemann_data, const Number p_star) const
    {
      const auto &[rho, u, p, gamma, a] = riemann_data;

      const auto factor =
          ScalarNumber(0.5) * (gamma + ScalarNumber(1.)) / gamma;

      const Number tmp = positive_part((p_star - p) / p);

      return u - a * std::sqrt(Number(1.) + factor * tmp);
    }

    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::lambda3_plus(const primitive_type &riemann_data,
                                             const Number p_star) const
    {
      const auto &[rho, u, p, gamma, a] = riemann_data;

      const auto factor =
          ScalarNumber(0.5) * (gamma + ScalarNumber(1.)) / gamma;

      const Number tmp = positive_part((p_star - p) / p);

      return u + a * std::sqrt(Number(1.) + factor * tmp);
    }

    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::compute_lambda(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        const Number p_star) const
    {
      const Number nu_11 = lambda1_minus(riemann_data_i, p_star);
      const Number nu_32 = lambda3_plus(riemann_data_j, p_star);

      return std::max(positive_part(nu_32), negative_part(nu_11));
    }

    template <int dim, typename Number>
    Number RiemannSolver<dim, Number>::compute(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;

      const Number p_min = std::min(p_i, p_j);
      const Number p_max = std::max(p_i, p_j);

      const Number phi_p_max = phi_of_p(riemann_data_i, riemann_data_j, p_max);
      const Number phi_p_min = phi_of_p(riemann_data_i, riemann_data_j, p_min);

      Number p_star_tilde = Number(0.);

      const Number p_star_RS = p_star_RS_aeos(riemann_data_i, riemann_data_j);

      const Number p_star_SS = p_star_SS_aeos(riemann_data_i, riemann_data_j);

      p_star_tilde = dealii::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          phi_p_max, Number(0.), p_star_RS, p_star_SS);

      p_star_tilde = dealii::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          phi_p_min, Number(0.), Number(0.), p_star_tilde);

      const Number p_2 =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              phi_p_max,
              Number(0.),
              p_star_tilde,
              std::min(p_max, p_star_tilde));

      return compute_lambda(riemann_data_i, riemann_data_j, p_2);
    }

  } // namespace EulerAEOS
} // namespace ryujin
