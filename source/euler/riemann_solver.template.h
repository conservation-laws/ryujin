//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "riemann_solver.h"

#include <newton.h>
#include <simd.h>

// #define DEBUG_RIEMANN_SOLVER

namespace ryujin
{
  namespace Euler
  {
    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::f(const primitive_type &riemann_data,
                                  const Number p_star) const
    {
      const auto view = hyperbolic_system.view<dim, Number>();
      const auto &gamma = view.gamma();

      const auto &[rho, u, p, a] = riemann_data;

      const Number Az = ScalarNumber(2.) / (rho * (gamma + Number(1.)));
      const Number Bz =
          (gamma - ScalarNumber(1.)) / (gamma + ScalarNumber(1.)) * p;
      const Number radicand = Az / (p_star + Bz);
      const Number true_value = (p_star - p) * std::sqrt(radicand);

      const auto exponent =
          ScalarNumber(0.5) * (gamma - ScalarNumber(1.)) / gamma;
      const Number factor = ryujin::pow(p_star / p, exponent) - Number(1.);
      const auto false_value =
          ScalarNumber(2.) * a * factor / (gamma - ScalarNumber(1.));

      return dealii::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_star, p, true_value, false_value);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::df(const primitive_type &riemann_data,
                                   const Number &p_star) const
    {
      const auto view = hyperbolic_system.view<dim, Number>();

      using ScalarNumber = typename get_value_type<Number>::type;
      const auto &gamma = view.gamma();
      const auto &gamma_inverse = view.gamma_inverse();
      const auto &gamma_minus_one_inverse = view.gamma_minus_one_inverse();
      const auto &gamma_plus_one_inverse = view.gamma_plus_one_inverse();

      const auto &[rho, u, p, a] = riemann_data;

      const Number radicand_inverse = ScalarNumber(0.5) * rho *
                                      ((gamma + ScalarNumber(1.)) * p_star +
                                       (gamma - ScalarNumber(1.)) * p);
      const Number denominator =
          (p_star + (gamma - ScalarNumber(1.)) * gamma_plus_one_inverse * p);
      const Number true_value =
          (denominator - ScalarNumber(0.5) * (p_star - p)) /
          (denominator * std::sqrt(radicand_inverse));

      const auto exponent =
          (ScalarNumber(-1.) - gamma) * ScalarNumber(0.5) * gamma_inverse;
      const Number factor = (gamma - ScalarNumber(1.)) * ScalarNumber(0.5) *
                            gamma_inverse * ryujin::pow(p_star / p, exponent) /
                            p;
      const auto false_value =
          factor * ScalarNumber(2.) * a * gamma_minus_one_inverse;

      return dealii::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_star, p, true_value, false_value);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::phi(const primitive_type &riemann_data_i,
                                    const primitive_type &riemann_data_j,
                                    const Number p_in) const
    {
      const Number &u_i = riemann_data_i[1];
      const Number &u_j = riemann_data_j[1];

      return f(riemann_data_i, p_in) + f(riemann_data_j, p_in) + u_j - u_i;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::dphi(const primitive_type &riemann_data_i,
                                     const primitive_type &riemann_data_j,
                                     const Number &p) const
    {
      return df(riemann_data_i, p) + df(riemann_data_j, p);
    }


    /*
     * The approximate Riemann solver is based on a function phi(p) that is
     * montone increasing in p, concave down and whose (weak) third
     * derivative is non-negative and locally bounded [1, p. 912]. Because we
     * actually do not perform any iteration for computing our wavespeed
     * estimate we can get away by only implementing a specialized variant of
     * the phi function that computes phi(p_max). It inlines the
     * implementation of the "f" function and eliminates all unnecessary
     * branches in "f".
     *
     * Cost: 0x pow, 2x division, 2x sqrt
     */
    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::phi_of_p_max(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto view = hyperbolic_system.view<dim, Number>();
      const auto &gamma = view.gamma();

      const auto &[rho_i, u_i, p_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, a_j] = riemann_data_j;

      const Number p_max = std::max(p_i, p_j);

      const Number radicand_inverse_i = ScalarNumber(0.5) * rho_i *
                                        ((gamma + ScalarNumber(1.)) * p_max +
                                         (gamma - ScalarNumber(1.)) * p_i);

      const Number value_i = (p_max - p_i) / std::sqrt(radicand_inverse_i);

      const Number radicand_inverse_j = ScalarNumber(0.5) * rho_j *
                                        ((gamma + ScalarNumber(1.)) * p_max +
                                         (gamma - ScalarNumber(1.)) * p_j);

      const Number value_j = (p_max - p_j) / std::sqrt(radicand_inverse_j);

      return value_i + value_j + u_j - u_i;
    }


    /*
     * Next we construct approximations for the two extreme wave speeds of
     * the Riemann fan [1, p. 912, (3.7) + (3.8)] and compute an upper bound
     * lambda_max of the maximal wave speed:
     */


    /*
     * see [1], page 912, (3.7)
     *
     * Cost: 0x pow, 1x division, 1x sqrt
     */
    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::lambda1_minus(
        const primitive_type &riemann_data, const Number p_star) const
    {
      const auto view = hyperbolic_system.view<dim, Number>();
      const auto &gamma = view.gamma();
      const auto &gamma_inverse = view.gamma_inverse();
      const auto factor =
          (gamma + ScalarNumber(1.0)) * ScalarNumber(0.5) * gamma_inverse;

      const auto &[rho, u, p, a] = riemann_data;
      const auto inv_p = ScalarNumber(1.0) / p;

      const Number tmp = positive_part((p_star - p) * inv_p);

      return u - a * std::sqrt(ScalarNumber(1.0) + factor * tmp);
    }


    /*
     * see [1], page 912, (3.8)
     *
     * Cost: 0x pow, 1x division, 1x sqrt
     */
    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::lambda3_plus(
        const primitive_type &primitive_state, const Number p_star) const
    {
      const auto view = hyperbolic_system.view<dim, Number>();
      const auto &gamma = view.gamma();
      const auto &gamma_inverse = view.gamma_inverse();
      const Number factor =
          (gamma + ScalarNumber(1.0)) * ScalarNumber(0.5) * gamma_inverse;

      const auto &[rho, u, p, a] = primitive_state;
      const auto inv_p = ScalarNumber(1.0) / p;

      const Number tmp = positive_part((p_star - p) * inv_p);
      return u + a * std::sqrt(Number(1.0) + factor * tmp);
    }


    /**
     * For two given primitive states <code>riemann_data_i</code> and
     * <code>riemann_data_j</code>, and two guesses p_1 <= p* <= p_2,
     * compute the gap in lambda between both guesses.
     *
     * See [1], page 914, (4.4a), (4.4b), (4.5), and (4.6)
     *
     * Cost: 0x pow, 4x division, 4x sqrt
     */
    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline std::array<Number, 2>
    RiemannSolver<dim, Number>::compute_gap(
        const std::array<Number, 4> &riemann_data_i,
        const std::array<Number, 4> &riemann_data_j,
        const Number p_1,
        const Number p_2)
    {
      const Number nu_11 = lambda1_minus(riemann_data_i, p_2 /*SIC!*/);
      const Number nu_12 = lambda1_minus(riemann_data_i, p_1 /*SIC!*/);

      const Number nu_31 = lambda3_plus(riemann_data_j, p_1);
      const Number nu_32 = lambda3_plus(riemann_data_j, p_2);

      const Number lambda_max =
          std::max(positive_part(nu_32), negative_part(nu_11));

      const Number gap =
          std::max(std::abs(nu_32 - nu_31), std::abs(nu_12 - nu_11));

      return {{gap, lambda_max}};
    }


    /*
     * For two given primitive states <code>riemann_data_i</code> and
     * <code>riemann_data_j</code>, and a guess p_2, compute an upper bound
     * for lambda.
     *
     * This is the same lambda_max as computed by compute_gap. The function
     * simply avoids a number of unnecessary computations (in case we do
     * not need to know the gap).
     *
     * Cost: 0x pow, 2x division, 2x sqrt
     */
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


    /*
     * Two-rarefaction approximation to p_star computed for two primitive
     * states <code>riemann_data_i</code> and <code>riemann_data_j</code>.
     *
     * See [1], page 914, (4.3)
     *
     * Cost: 2x pow, 2x division, 0x sqrt
     */
    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::p_star_two_rarefaction(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto view = hyperbolic_system.view<dim, Number>();
      const auto &gamma = view.gamma();
      const auto &gamma_inverse = view.gamma_inverse();
      const auto &gamma_minus_one_inverse = view.gamma_minus_one_inverse();

      const auto &[rho_i, u_i, p_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, a_j] = riemann_data_j;
      const auto inv_p_j = ScalarNumber(1.) / p_j;

      /*
       * Nota bene (cf. [1, (4.3)]):
       *   a_Z^0 * sqrt(1 - b * rho_Z) = a_Z * (1 - b * rho_Z)
       * We have computed a_Z already, so we are simply going to use this
       * identity below:
       */

      const auto factor = (gamma - ScalarNumber(1.)) * ScalarNumber(0.5);

      /*
       * Nota bene (cf. [1, (3.6)]: The condition "numerator > 0" is the
       * well-known non-vacuum condition. In case we encounter numerator <= 0
       * then p_star = 0 is the correct pressure to compute the wave speed.
       * Therefore, all we have to do is to take the positive part of the
       * expression:
       */

      const Number numerator = positive_part(a_i + a_j - factor * (u_j - u_i));
      const Number denominator =
          a_i * ryujin::pow(p_i * inv_p_j, -factor * gamma_inverse) + a_j;

      const auto exponent = ScalarNumber(2.0) * gamma * gamma_minus_one_inverse;

      const auto p_1_tilde =
          p_j * ryujin::pow(numerator / denominator, exponent);

#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << "p_star_two_rarefaction = " << p_1_tilde << std::endl;
#endif
      return p_1_tilde;
    }


    /*
     * Failsafe approximation to p_star computed for two primitive
     * states <code>riemann_data_i</code> and <code>riemann_data_j</code>.
     *
     * See [1], page 914, (4.3)
     *
     * Cost: 2x pow, 2x division, 0x sqrt
     */
    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    RiemannSolver<dim, Number>::p_star_failsafe(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto view = hyperbolic_system.view<dim, Number>();
      const auto &gamma = view.gamma();

      const auto &[rho_i, u_i, p_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, a_j] = riemann_data_j;

      /*
       * Compute (5.11) formula for \tilde p_2^\ast:
       *
       * Cost: 0x pow, 3x division, 3x sqrt
       */

      const Number p_max = std::max(p_i, p_j);

      Number radicand_i = ScalarNumber(2.) * p_max;
      radicand_i /=
          rho_i * ((gamma + Number(1.)) * p_max + (gamma - Number(1.)) * p_i);

      const Number x_i = std::sqrt(radicand_i);

      Number radicand_j = ScalarNumber(2.) * p_max;
      radicand_j /=
          rho_j * ((gamma + Number(1.)) * p_max + (gamma - Number(1.)) * p_j);

      const Number x_j = std::sqrt(radicand_j);

      const Number a = x_i + x_j;
      const Number b = u_j - u_i;
      const Number c = -p_i * x_i - p_j * x_j;

      const Number base = (-b + std::sqrt(b * b - ScalarNumber(4.) * a * c)) /
                          (ScalarNumber(2.) * a);
      const Number p_2_tilde = base * base;

#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << "p_star_failsafe = " << p_2_tilde << std::endl;
#endif
      return p_2_tilde;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    RiemannSolver<dim, Number>::riemann_data_from_state(
        const state_type &U, const dealii::Tensor<1, dim, Number> &n_ij) const
        -> primitive_type
    {
      const auto view = hyperbolic_system.view<dim, Number>();

      const auto rho = view.density(U);
      const auto rho_inverse = Number(1.0) / rho;

      const auto m = view.momentum(U);
      const auto proj_m = n_ij * m;
      const auto perp = m - proj_m * n_ij;

      const auto E =
          view.total_energy(U) - Number(0.5) * perp.norm_square() * rho_inverse;

      using state_type_1d =
          typename HyperbolicSystemView<1, Number>::state_type;
      const auto view_1d = hyperbolic_system.view<1, Number>();

      const auto state = state_type_1d{{rho, proj_m, E}};
      const auto p = view_1d.pressure(state);
      const auto a = view_1d.speed_of_sound(state);
      return {{rho, proj_m * rho_inverse, p, a}};
    }


    template <int dim, typename Number>
    Number RiemannSolver<dim, Number>::compute(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      /*
       * For exactly solving the Riemann problem we need to start with a
       * good upper and lower bound, p_1 <= p_star <= p_2, for finding
       * phi(p_star) == 0. This implies that we have to ensure that
       * phi(p_2) >= 0 and phi(p_1) <= 0.
       *
       * Instead of solving the Riemann problem exactly, however we will
       * simply use the upper bound p_2 (with p_2 >= p_star) to compute
       * lambda_max and return the estimate.
       *
       * We will use three candidates, p_min, p_max and the two rarefaction
       * approximation p_star_tilde. We have (up to round-off errors) that
       * phi(p_star_tilde) >= 0. So this is a safe upper bound, it might
       * just be too large.
       *
       * Depending on the sign of phi(p_max) we select the following ranges:
       *
       *   phi(p_max) <  0:
       *     p_1  <-  p_max   and   p_2  <-  p_star_tilde
       *
       *   phi(p_max) >= 0:
       *     p_1  <-  p_min   and   p_2  <-  min(p_max, p_star_tilde)
       *
       * Nota bene:
       *
       *  - The special case phi(p_max) == 0 as discussed in [1] is already
       *    contained in the second condition.
       *
       *  - In principle, we would have to treat the case phi(p_min) > 0 as
       *    well. This corresponds to two expansion waves and a good
       *    estimate for the wavespeed is obtained by simply computing
       *    lambda_max with p_2 = 0.
       *
       *    However, it turns out that numerically in this case we will
       *    have
       *
       *      0 < p_star <= p_star_tilde <= p_min <= p_max.
       *
       *    So it is sufficient to end up with p_2 = p_star_tilde (!!) to
       *    compute the exact same wave speed as for p_2 = 0.
       *
       *    Note: If for some reason p_star should be computed exactly,
       *    then p_1 has to be set to zero. This can be done efficiently by
       *    simply checking for p_2 < p_1 and setting p_1 <- 0 if
       *    necessary.
       */

      const auto &[rho_i, u_i, p_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, a_j] = riemann_data_j;

#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << "rho_left: " << rho_i << std::endl;
      std::cout << "u_left: " << u_i << std::endl;
      std::cout << "p_left: " << p_i << std::endl;
      std::cout << "a_left: " << a_i << std::endl;
      std::cout << "rho_right: " << rho_j << std::endl;
      std::cout << "u_right: " << u_j << std::endl;
      std::cout << "p_right: " << p_j << std::endl;
      std::cout << "a_right: " << a_j << std::endl;
#endif

      const Number p_max = std::max(p_i, p_j);

      const Number rarefaction =
          p_star_two_rarefaction(riemann_data_i, riemann_data_j);
      const Number failsafe = p_star_failsafe(riemann_data_i, riemann_data_j);
      const Number p_star_tilde = std::min(rarefaction, failsafe);

      const Number phi_p_max = phi_of_p_max(riemann_data_i, riemann_data_j);

      const Number p_2 =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              phi_p_max,
              Number(0.),
              p_star_tilde,
              std::min(p_max, p_star_tilde));

#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << "   p^*_tilde  = " << p_2 << "\n";
      std::cout << "   phi(p_*_t) = "
                << phi(riemann_data_i, riemann_data_j, p_2) << "\n";
      std::cout << "-> lambda_max = "
                << compute_lambda(riemann_data_i, riemann_data_j, p_2)
                << std::endl;
#endif

      return compute_lambda(riemann_data_i, riemann_data_j, p_2);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number RiemannSolver<dim, Number>::compute(
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

  } // namespace Euler
} // namespace ryujin
