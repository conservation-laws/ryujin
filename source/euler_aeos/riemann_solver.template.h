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
  /* This is a modification of the RiemannSolver Class in ryujin/euler. We refer
   * the reader to:

   * Invariant Domain-Preserving Approximations for the Euler Equations with
   * Tabulated Equation of State Bennett Clayton, Jean-Luc Guermond, and Bojan *
   * Popov, SIAM Journal on Scientific Computing 2022 44:1, A444-A470
   *
   * for more details. More documentation coming.
   */

  /* A namespace for helper functions -- Maybe this can be written better? */
  namespace helper_functions
  {
    /* Compute \alpha_Z function */
    template <typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    compute_alpha(const std::array<Number, 4> &riemann_data,
                  const Number b_interp)
    {
      using ScalarNumber = typename get_value_type<Number>::type;

      const auto &[rho, v, p, gamma_] = riemann_data;
      const Number x = Number(1.) - b_interp * rho;
      const Number a = std::sqrt(gamma_ * p / (rho * x));

      const Number numerator =
          ScalarNumber(2.) * a * (Number(1.) - b_interp * rho);

      const Number denominator = gamma_ - Number(1.);

      return numerator / denominator;
    }

    /* Compute c(\gamma_Z) function */
    template <typename Number>
    DEAL_II_ALWAYS_INLINE inline Number compute_c(const Number gamma_Z)
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
    template <typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    p_star_RS_aeos(const std::array<Number, 4> &riemann_data_i,
                   const std::array<Number, 4> &riemann_data_j,
                   const Number b_interp)
    {
      using ScalarNumber = typename get_value_type<Number>::type;

      const auto &[rho_i, u_i, p_i, gamma_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j] = riemann_data_j;

      /* First get p_min, p_max. Then, we get rho_min, rho_max, gamma_min,
       * gamma_max. Note: gamma_min is associated with p_min and is technically
       * not the minimum of gamma_left vs gamma_right. */

      const Number p_min = std::min(p_i, p_j);
      const Number p_max = std::max(p_i, p_j);

      const Number rho_min =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, rho_i, rho_j);

      const Number rho_max = dealii::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_i, p_j, rho_i, rho_j);

      const Number gamma_min =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, gamma_i, gamma_j);

      const Number gamma_max = dealii::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_i, p_j, gamma_i, gamma_j);

      /* Compute auxiliary quantities needed for p_star_RS */
      const std::array<Number, 4> min_values = {
          rho_min, Number(0.), p_min, gamma_min};

      const std::array<Number, 4> max_values = {
          rho_max, Number(0.), p_max, gamma_max};

      const Number alpha_min = compute_alpha(min_values, b_interp);
      const Number alpha_max = compute_alpha(max_values, b_interp);
      const Number c_gamma_min = compute_c(gamma_min);

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

    /* For p_star in the Shock/Shock case. Normal approximation */
    template <typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    p_star_SS_aeos(const std::array<Number, 4> &riemann_data_i,
                   const std::array<Number, 4> &riemann_data_j,
                   const Number b_interp)
    {
      using ScalarNumber = typename get_value_type<Number>::type;

      const auto &[rho_i, u_i, p_i, gamma_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j] = riemann_data_j;

      /* First get gamma_m */
      const Number gamma_m = std::min(gamma_i, gamma_j);

      /* Compute alpha_hat_left and alpha_hat_right  */
      const Number alpha_hat_left =
          compute_c(gamma_i) * compute_alpha(riemann_data_i, b_interp);
      const Number alpha_hat_right =
          compute_c(gamma_j) * compute_alpha(riemann_data_j, b_interp);

      const Number exp = (gamma_m - Number(1.)) / (ScalarNumber(2.) * gamma_m);
      const Number exp_inv = Number(1.) / exp;

      /* Then we can compute p_star_SS */
      const Number numerator = alpha_hat_left + alpha_hat_right - (u_j - u_i);

      const Number denominator = alpha_hat_left * ryujin::vec_pow(p_i, -exp) +
                                 alpha_hat_right * ryujin::vec_pow(p_j, -exp);

      const Number base = numerator / denominator;

      return ryujin::vec_pow(base, exp_inv);
    }
  } // namespace helper_functions

  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  RiemannSolver<dim, Number>::f(const primitive_type &primitive_state,
                                const Number p_star) const
  {
    using ScalarNumber = typename get_value_type<Number>::type;

    const auto &[rho, u, p, gamma_] = primitive_state;

    const Number g_minus_one_ = gamma_ - Number(1.);

    const Number one_minus_b_rho = Number(1.) - ScalarNumber(b_interp) * rho;

    const Number a = std::sqrt(gamma_ * p / (rho * one_minus_b_rho));

    Number radicand_inverse =
        ScalarNumber(0.5) * rho *
        ((gamma_ + Number(1.)) * p_star + g_minus_one_ * p) / one_minus_b_rho;

    const Number true_value = (p_star - p) / std::sqrt(radicand_inverse);

    const auto exponent = ScalarNumber(0.5) * g_minus_one_ / gamma_;
    const Number factor = ryujin::vec_pow(p_star / p, exponent) - Number(1.);
    const auto false_value =
        ScalarNumber(2.) * a * one_minus_b_rho * factor / g_minus_one_;

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
  RiemannSolver<dim, Number>::lambda1_minus(const primitive_type &riemann_data,
                                            const Number p_star) const
  {
    using ScalarNumber = typename get_value_type<Number>::type;

    const auto &[rho, u, p, gamma_] = riemann_data;

    const auto factor =
        ScalarNumber(0.5) * (gamma_ + ScalarNumber(1.)) / gamma_;

    const Number tmp = positive_part((p_star - p) / p);

    const Number a_denominator =
        rho * (Number(1.) - ScalarNumber(b_interp) * rho);

    const Number a = std::sqrt(gamma_ * p / a_denominator);

    return u - a * std::sqrt(Number(1.) + factor * tmp);
  }

  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  RiemannSolver<dim, Number>::lambda3_plus(const primitive_type &riemann_data,
                                           const Number p_star) const
  {
    using ScalarNumber = typename get_value_type<Number>::type;

    const auto &[rho, u, p, gamma_] = riemann_data;

    const auto factor =
        ScalarNumber(0.5) * (gamma_ + ScalarNumber(1.)) / gamma_;

    const Number tmp = positive_part((p_star - p) / p);

    const Number a_denominator =
        rho * (Number(1.) - ScalarNumber(b_interp) * rho);

    const Number a = std::sqrt(gamma_ * p / a_denominator);

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
    using namespace helper_functions;

    /* First we get left and right states */
    const auto &[rho_i, u_i, p_i, gamma_i] = riemann_data_i;
    const auto &[rho_j, u_j, p_j, gamma_j] = riemann_data_j;

    /* First get p_min, p_max */
    const Number p_min = std::min(p_i, p_j);
    const Number p_max = std::max(p_i, p_j);

    const Number phi_p_max = phi_of_p(riemann_data_i, riemann_data_j, p_max);

    const Number phi_p_min = phi_of_p(riemann_data_i, riemann_data_j, p_min);

    Number p_star_tilde = Number(0.);

    const Number p_star_RS =
        p_star_RS_aeos(riemann_data_i, riemann_data_j, Number(b_interp));

    const Number p_star_SS =
        p_star_SS_aeos(riemann_data_i, riemann_data_j, Number(b_interp));

    p_star_tilde = dealii::compare_and_apply_mask<
        dealii::SIMDComparison::greater_than_or_equal>(
        phi_p_max, Number(0.), p_star_RS, p_star_SS);

    p_star_tilde = dealii::compare_and_apply_mask<
        dealii::SIMDComparison::greater_than_or_equal>(
        phi_p_min, Number(0.), Number(0.), p_star_tilde);

    const Number p_2 =
        dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
            phi_p_max, Number(0.), p_star_tilde, std::min(p_max, p_star_tilde));

    return compute_lambda(riemann_data_i, riemann_data_j, p_2);
  }


} /* namespace ryujin */
