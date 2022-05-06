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
  using namespace dealii;

  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  RiemannSolver<dim, Number>::f(const primitive_type &riemann_data,
                                const Number &h_Z) const
  {
    using ScalarNumber = typename get_value_type<Number>::type;

    const auto &[h, u, a] = riemann_data;

    const auto left_value = ScalarNumber(2.) * (a - std::sqrt(gravity * h_Z));

    const Number radicand = ScalarNumber(0.5) * gravity * (h + h_Z) / (h * h_Z);
    const Number right_value = (h - h_Z) * std::sqrt(radicand);

    return dealii::compare_and_apply_mask<
        dealii::SIMDComparison::greater_than_or_equal>(
        h_Z, h, left_value, right_value);
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  RiemannSolver<dim, Number>::phi(const primitive_type &riemann_data_i,
                                  const primitive_type &riemann_data_j,
                                  const Number &h) const
  {
    const Number &u_i = riemann_data_i[1];
    const Number &u_j = riemann_data_j[1];

    return f(riemann_data_i, h) + f(riemann_data_j, h) + u_j - u_i;
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  RiemannSolver<dim, Number>::lambda1_minus(const primitive_type &riemann_data,
                                            const Number h_star) const
  {
    using ScalarNumber = typename get_value_type<Number>::type;

    const auto &[h, u, a] = riemann_data;

    const Number factor = positive_part((h_star - h) / h);
    const Number half_factor = ScalarNumber(0.5) * factor;

    return u - a * std::sqrt((ScalarNumber(1.) + half_factor) *
                             (ScalarNumber(1.) + factor));
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  RiemannSolver<dim, Number>::lambda3_plus(const primitive_type &riemann_data,
                                           const Number h_star) const
  {
    using ScalarNumber = typename get_value_type<Number>::type;

    const auto &[h, u, a] = riemann_data;

    const Number factor = positive_part((h_star - h) / h);
    const Number half_factor = ScalarNumber(0.5) * factor;

    return u + a * std::sqrt((ScalarNumber(1.) + half_factor) *
                             (ScalarNumber(1.) + factor));
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  RiemannSolver<dim, Number>::compute_lambda(
      const primitive_type &riemann_data_i,
      const primitive_type &riemann_data_j,
      const Number h_star) const
  {
    const Number lambda1 = lambda1_minus(riemann_data_i, h_star);
    const Number lambda3 = lambda3_plus(riemann_data_j, h_star);

    return std::max(negative_part(lambda1), positive_part(lambda3));
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  RiemannSolver<dim, Number>::h_star_two_rarefaction(
      const primitive_type &riemann_data_i,
      const primitive_type &riemann_data_j) const
  {
    using ScalarNumber = typename get_value_type<Number>::type;

    const auto &[h_i, u_i, a_i] = riemann_data_i;
    const auto &[h_j, u_j, a_j] = riemann_data_j;

    const Number h_min = std::min(h_i, h_j);
    const Number h_max = std::max(h_i, h_j);

    const Number a_min = std::sqrt(gravity * h_min);
    const Number a_max = std::sqrt(gravity * h_max);

    const Number sqrt_two = std::sqrt(ScalarNumber(2.));

    /* x0 = (2 sqrt(2) - 1)^2 */
    const Number x0 = Number(9.) - ScalarNumber(4.) * sqrt_two;

    const Number phi_value_min =
        phi(riemann_data_i, riemann_data_j, x0 * h_min);

    const Number phi_value_max =
        phi(riemann_data_i, riemann_data_j, x0 * h_max);

    /* We compute the three h_star quantities */

    // h_star_left
    Number tmp = positive_part(u_i - u_j + ScalarNumber(2.) * (a_i + a_j));
    const Number h_star_left =
        ScalarNumber(0.0625) * gravity_inverse * tmp * tmp;

    // h_star_middle
    tmp = Number(1.) + sqrt_two * (u_i - u_j) / (a_min + a_max);
    const Number h_star_middle = std::sqrt(h_min * h_max) * tmp;

    // h_star_right
    const auto left_radicand =
        ScalarNumber(3.) * h_min +
        ScalarNumber(2.) * sqrt_two * std::sqrt(h_min * h_max);

    const auto right_radicand =
        sqrt_two * std::sqrt(gravity_inverse * h_min) * (u_i - u_j);

    tmp = std::sqrt(positive_part(left_radicand + right_radicand));
    tmp -= sqrt_two * std::sqrt(h_min);

    const Number h_star_right = tmp * tmp;

    /* Finally define h_star */
    Number h_star = dealii::compare_and_apply_mask<
        dealii::SIMDComparison::less_than_or_equal>(
        Number(0.), phi_value_min, h_star_left, h_star_right);

    h_star = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
        phi_value_max, Number(0.), h_star_middle, h_star_right);

    return h_star;
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline auto
  RiemannSolver<dim, Number>::riemann_data_from_state(
      const HyperbolicSystem &hyperbolic_system,
      const state_type &U,
      const dealii::Tensor<1, dim, Number> &n_ij) const -> primitive_type
  {
    const auto h = std::max(hyperbolic_system.water_depth(U), Number(h_tiny));

    const auto vel = hyperbolic_system.momentum(U) *
                     hyperbolic_system.inverse_water_depth(U);
    const auto proj_vel = n_ij * vel;

    const auto a = std::sqrt(h * ScalarNumber(gravity));

    return {{h, proj_vel, a}};
  }


  template <int dim, typename Number>
  inline Number RiemannSolver<dim, Number>::compute(
      const primitive_type &riemann_data_i,
      const primitive_type &riemann_data_j) const
  {
    const Number h_star =
        h_star_two_rarefaction(riemann_data_i, riemann_data_j);

    const Number lambda_max =
        compute_lambda(riemann_data_i, riemann_data_j, h_star);

    return lambda_max;
  }


  template <int dim, typename Number>
  Number RiemannSolver<dim, Number>::compute(
      const state_type &U_i,
      const state_type &U_j,
      const dealii::Tensor<1, dim, Number> &n_ij) const
  {
    const auto riemann_data_i =
        riemann_data_from_state(hyperbolic_system, U_i, n_ij);
    const auto riemann_data_j =
        riemann_data_from_state(hyperbolic_system, U_j, n_ij);
    return compute(riemann_data_i, riemann_data_j);
  }


} /* namespace ryujin */
