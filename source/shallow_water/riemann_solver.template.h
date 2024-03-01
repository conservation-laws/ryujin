//
// SPDX-License-Identifier: BSD-3-Clause
// [LANL Copyright Statement]
// Copyright (C) 2022 - 2024 by the ryujin authors
// Copyright (C) 2023 - 2024 by Triad National Security, LLC
//

#pragma once

#include <compile_time_options.h>

#include "riemann_solver.h"

#include <newton.h>
#include <simd.h>

// #define DEBUG_RIEMANN_SOLVER

namespace ryujin
{
  namespace ShallowWater
  {
    using namespace dealii;

    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::f(const primitive_type &riemann_data_Z,
                                  const Number &h) const
    {
      const auto view = hyperbolic_system.view<dim, Number>();
      const ScalarNumber gravity = view.gravity();

      const auto &[h_Z, u_Z, a_Z] = riemann_data_Z;

      const auto left_value = ScalarNumber(2.) * (std::sqrt(gravity * h) - a_Z);

      const Number radicand =
          ScalarNumber(0.5) * gravity * (h + h_Z) / (h * h_Z);
      const Number right_value = (h - h_Z) * std::sqrt(radicand);

      return dealii::compare_and_apply_mask<
          dealii::SIMDComparison::less_than_or_equal>(
          h, h_Z, left_value, right_value);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::phi(const primitive_type &riemann_data_i,
                                    const primitive_type &riemann_data_j,
                                    const Number &h) const
    {
      const Number &u_i = riemann_data_i[1];
      const Number &u_j = riemann_data_j[1];

#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << "f_L --> " << f(riemann_data_i, h) << std::endl;
      std::cout << "f_R --> " << f(riemann_data_j, h) << std::endl;
#endif
      return f(riemann_data_i, h) + f(riemann_data_j, h) + u_j - u_i;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::lambda1_minus(
        const primitive_type &riemann_data, const Number h_star) const
    {
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
    RiemannSolver<dim, Number>::compute_h_star(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto view = hyperbolic_system.view<dim, Number>();
      const ScalarNumber gravity = view.gravity();
      const auto gravity_inverse = ScalarNumber(1.) / gravity;

      const auto &[h_i, u_i, a_i] = riemann_data_i;
      const auto &[h_j, u_j, a_j] = riemann_data_j;

      const Number h_min = std::min(h_i, h_j);
      const Number h_max = std::max(h_i, h_j);

#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << h_min << "  <- h_min/max ->  " << h_max << std::endl;
#endif

      const Number a_min = std::sqrt(gravity * h_min);
      const Number a_max = std::sqrt(gravity * h_max);

#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << a_min << "  <- a_min/max ->  " << a_max << std::endl;
#endif

      const Number sqrt_two = std::sqrt(ScalarNumber(2.));

      /* x0 = (2 sqrt(2) - 1)^2 */
      const Number x0 = Number(9.) - ScalarNumber(4.) * sqrt_two;

      const Number phi_value_min =
          phi(riemann_data_i, riemann_data_j, x0 * h_min);
#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << "phi_value_min ->" << phi_value_min << std::endl;
#endif

      const Number phi_value_max =
          phi(riemann_data_i, riemann_data_j, x0 * h_max);
#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << "phi_value_max ->" << phi_value_max << std::endl;
#endif

      /* We compute the three h_star quantities */

      Number tmp;

      /* Double rarefaction case (h_star left): */

      tmp = positive_part(u_i - u_j + ScalarNumber(2.) * (a_i + a_j));
      const Number h_star_left =
          ScalarNumber(0.0625) * gravity_inverse * tmp * tmp;

#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << "left: " << h_star_left << std::endl;
#endif

      /* Double modified shock (h_star middle): */

      tmp = Number(1.) + sqrt_two * (u_i - u_j) / (a_min + a_max);
      const Number h_star_middle = std::sqrt(h_min * h_max) * tmp;

#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << "middle: " << h_star_middle << std::endl;
#endif

      /* Expansion and modified shock (h_star right): */

      const auto left_radicand =
          ScalarNumber(3.) * h_min +
          ScalarNumber(2.) * sqrt_two * std::sqrt(h_min * h_max);

      const auto right_radicand =
          sqrt_two * std::sqrt(gravity_inverse * h_min) * (u_i - u_j);

      tmp = std::sqrt(positive_part(left_radicand + right_radicand));
      tmp -= sqrt_two * std::sqrt(h_min);

      const Number h_star_right = tmp * tmp;

#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << "right: " << h_star_right << std::endl;
#endif

      /* Finally define h_star */

      Number h_star = dealii::compare_and_apply_mask<
          dealii::SIMDComparison::less_than_or_equal>(
          Number(0.), phi_value_min, h_star_left, h_star_right);

      h_star =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              phi_value_max, Number(0.), h_star_middle, h_star_right);

      return h_star;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    RiemannSolver<dim, Number>::riemann_data_from_state(
        const state_type &U, const dealii::Tensor<1, dim, Number> &n_ij) const
        -> primitive_type
    {
      const auto view = hyperbolic_system.view<dim, Number>();

      const Number h = view.water_depth_sharp(U);
      const Number gravity = view.gravity();

      const auto velocity = view.momentum(U) / h;
      const auto projected_velocity = n_ij * velocity;
      const auto a = std::sqrt(h * gravity);

      return {{h, projected_velocity, a}};
    }


    template <int dim, typename Number>
    inline Number RiemannSolver<dim, Number>::compute(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const Number h_star = compute_h_star(riemann_data_i, riemann_data_j);

      const Number lambda_max =
          compute_lambda(riemann_data_i, riemann_data_j, h_star);

      return lambda_max;
    }


    template <int dim, typename Number>
    Number RiemannSolver<dim, Number>::compute(
        const state_type &U_i,
        const state_type &U_j,
        const unsigned int /*i*/,
        const unsigned int * /*js*/,
        const dealii::Tensor<1, dim, Number> &n_ij) const
    {
      const auto riemann_data_i = riemann_data_from_state(U_i, n_ij);
      const auto riemann_data_j = riemann_data_from_state(U_j, n_ij);
      return compute(riemann_data_i, riemann_data_j);
    }

  } // namespace ShallowWater
} // namespace ryujin
