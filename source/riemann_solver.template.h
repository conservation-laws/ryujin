//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "limiter.h"
#include "newton.h"
#include "riemann_solver.h"
#include "simd.h"

namespace ryujin
{
  using namespace dealii;

  /*
   * We construct a function phi(p) that is montone increasing in p,
   * concave down and whose (weak) third derivative is non-negative and
   * locally bounded [1, p. 912]. We also need to implement derivatives
   * of phi for the quadratic Newton search:
   */

  /**
   * See [1], page 912, (3.4).
   *
   * Cost: 1x pow, 1x division, 2x sqrt
   */
  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  RiemannSolver<dim, Number>::f(const std::array<Number, 4> &primitive_state,
                                const Number &p_star)
  {
    using ScalarNumber = typename get_value_type<Number>::type;

    const auto &[rho, u, p, a] = primitive_state;

    const Number radicand_inverse =
        ScalarNumber(0.5) * rho *
        ((gamma + ScalarNumber(1.)) * p_star + (gamma - ScalarNumber(1.)) * p);
    const Number true_value = (p_star - p) / std::sqrt(radicand_inverse);

    const auto exponent =
        (gamma - ScalarNumber(1.)) * ScalarNumber(0.5) * gamma_inverse;
    const Number factor = ryujin::pow(p_star / p, exponent) - Number(1.);
    const auto false_value =
        factor * ScalarNumber(2.) * a * gamma_minus_one_inverse;

    return dealii::compare_and_apply_mask<
        dealii::SIMDComparison::greater_than_or_equal>(
        p_star, p, true_value, false_value);
  }

  /**
   * See [1], page 912, (3.4).
   *
   * Cost: 1x pow, 3x division, 1x sqrt
   */
  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  RiemannSolver<dim, Number>::df(const std::array<Number, 4> &primitive_state,
                                 const Number &p_star)
  {
    using ScalarNumber = typename get_value_type<Number>::type;

    const auto &[rho, u, p, a] = primitive_state;

    const Number radicand_inverse =
        ScalarNumber(0.5) * rho *
        ((gamma + ScalarNumber(1.)) * p_star + (gamma - ScalarNumber(1.)) * p);
    const Number denominator =
        (p_star + (gamma - ScalarNumber(1.)) * gamma_plus_one_inverse * p);
    const Number true_value = (denominator - ScalarNumber(0.5) * (p_star - p)) /
                              (denominator * std::sqrt(radicand_inverse));

    const auto exponent =
        (ScalarNumber(-1.) - gamma) * ScalarNumber(0.5) * gamma_inverse;
    const Number factor = (gamma - ScalarNumber(1.)) * ScalarNumber(0.5) *
                          gamma_inverse * ryujin::pow(p_star / p, exponent) / p;
    const auto false_value =
        factor * ScalarNumber(2.) * a * gamma_minus_one_inverse;

    return dealii::compare_and_apply_mask<
        dealii::SIMDComparison::greater_than_or_equal>(
        p_star, p, true_value, false_value);
  }


  /**
   * See [1], page 912, (3.3).
   *
   * Cost: 2x pow, 2x division, 4x sqrt
   */
  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  RiemannSolver<dim, Number>::phi(const std::array<Number, 4> &riemann_data_i,
                                  const std::array<Number, 4> &riemann_data_j,
                                  const Number &p)
  {
    const Number &u_i = riemann_data_i[1];
    const Number &u_j = riemann_data_j[1];

    return f(riemann_data_i, p) + f(riemann_data_j, p) + u_j - u_i;
  }


  /**
   * This is a specialized variant of phi() that computes phi(p_max). It
   * inlines the implementation of f() and eliminates all unnecessary
   * branches in f().
   *
   * Cost: 0x pow, 2x division, 2x sqrt
   */
  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number RiemannSolver<dim, Number>::phi_of_p_max(
      const std::array<Number, 4> &riemann_data_i,
      const std::array<Number, 4> &riemann_data_j)
  {
    using ScalarNumber = typename get_value_type<Number>::type;

    const auto &[rho_i, u_i, p_i, a_i] = riemann_data_i;
    const auto &[rho_j, u_j, p_j, a_j] = riemann_data_j;

    const Number p_max = std::max(p_i, p_j);

    const Number radicand_inverse_i =
        ScalarNumber(0.5) * rho_i *
        ((gamma + ScalarNumber(1.)) * p_max + (gamma - ScalarNumber(1.)) * p_i);

    const Number value_i = (p_max - p_i) / std::sqrt(radicand_inverse_i);

    const Number radicand_inverse_j =
        ScalarNumber(0.5) * rho_j *
        ((gamma + ScalarNumber(1.)) * p_max + (gamma - ScalarNumber(1.)) * p_j);

    const Number value_j = (p_max - p_j) / std::sqrt(radicand_inverse_j);

    return value_i + value_j + u_j - u_i;
  }


  /**
   * See [1], page 912, (3.3).
   *
   * Cost: 2x pow, 6x division, 2x sqrt
   */
  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  RiemannSolver<dim, Number>::dphi(const std::array<Number, 4> &riemann_data_i,
                                   const std::array<Number, 4> &riemann_data_j,
                                   const Number &p)
  {
    return df(riemann_data_i, p) + df(riemann_data_j, p);
  }


  /*
   * Next we construct approximations for the two extreme wave speeds of
   * the Riemann fan [1, p. 912, (3.7) + (3.8)] and compute a gap (based
   * on the quality of our current approximation of the two wave speeds)
   * and an upper bound lambda_max of the maximal wave speed:
   */


  /**
   * see [1], page 912, (3.7)
   *
   * Cost: 0x pow, 1x division, 1x sqrt
   */
  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number RiemannSolver<dim, Number>::lambda1_minus(
      const std::array<Number, 4> &riemann_data, const Number p_star)
  {
    using ScalarNumber = typename get_value_type<Number>::type;

    const auto &[rho, u, p, a] = riemann_data;

    const auto factor =
        (gamma + ScalarNumber(1.0)) * ScalarNumber(0.5) * gamma_inverse;
    const Number tmp = positive_part((p_star - p) / p);

    return u - a * std::sqrt(Number(1.0) + factor * tmp);
  }


  /**
   * see [1], page 912, (3.8)
   *
   * Cost: 0x pow, 1x division, 1x sqrt
   */
  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number RiemannSolver<dim, Number>::lambda3_plus(
      const std::array<Number, 4> &primitive_state, const Number p_star)
  {
    using ScalarNumber = typename get_value_type<Number>::type;

    const auto &[rho, u, p, a] = primitive_state;

    const Number factor =
        (gamma + ScalarNumber(1.0)) * ScalarNumber(0.5) * gamma_inverse;
    const Number tmp = positive_part((p_star - p) / p);
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


  /**
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
      const std::array<Number, 4> &riemann_data_i,
      const std::array<Number, 4> &riemann_data_j,
      const Number p_star)
  {
    const Number nu_11 = lambda1_minus(riemann_data_i, p_star);
    const Number nu_32 = lambda3_plus(riemann_data_j, p_star);

    return std::max(positive_part(nu_32), negative_part(nu_11));
  }


  /**
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
      const std::array<Number, 4> &riemann_data_i,
      const std::array<Number, 4> &riemann_data_j)
  {
    using ScalarNumber = typename get_value_type<Number>::type;

    const auto &[rho_i, u_i, p_i, a_i] = riemann_data_i;
    const auto &[rho_j, u_j, p_j, a_j] = riemann_data_j;

    /*
     * Nota bene (cf. [1, (4.3)]):
     *   a_Z^0 * sqrt(1 - b * rho_Z) = a_Z * (1 - b * rho_Z)
     * We have computed a_Z already, so we are simply going to use this
     * identity below:
     */

    const auto factor = (gamma - ScalarNumber(1.)) * ScalarNumber(0.5);

    const Number numerator = a_i + a_j - factor * (u_j - u_i);
    const Number denominator =
        a_i * ryujin::pow(p_i / p_j, -factor * gamma_inverse) + a_j;

    const auto exponent = ScalarNumber(2.0) * gamma * gamma_minus_one_inverse;

    return p_j * ryujin::pow(numerator / denominator, exponent);
  }


  /**
   * Given the pressure minimum and maximum and two corresponding
   * densities we compute approximations for the density of corresponding
   * shock and expansion waves.
   *
   * [2] Formula (4.4)
   *
   * Cost: 2x pow, 2x division, 0x sqrt
   */
  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline std::array<Number, 4>
  RiemannSolver<dim, Number>::shock_and_expansion_density(
      const Number p_min,
      const Number p_max,
      const Number rho_p_min,
      const Number rho_p_max,
      const Number p_1,
      const Number p_2)
  {
    const auto gm1_gp2 = gamma_minus_one_over_gamma_plus_one;

    const auto rho_p_min_shk =
        rho_p_min * (gm1_gp2 * p_min + p_1) / (gm1_gp2 * p_1 + p_min);

    const auto rho_p_max_shk =
        rho_p_min * (gm1_gp2 * p_max + p_1) / (gm1_gp2 * p_1 + p_max);

    const auto rho_p_min_exp =
        rho_p_min * ryujin::pow(p_2 / p_min, gamma_inverse);

    const auto rho_p_max_exp =
        rho_p_max * ryujin::pow(p_2 / p_max, gamma_inverse);

    return {{rho_p_min_shk, rho_p_max_shk, rho_p_min_exp, rho_p_max_exp}};
  }

  /**
   * For a given (2+dim dimensional) state vector <code>U</code>, and a
   * (normalized) "direction" n_ij, first compute the corresponding
   * projected state in the corresponding 1D Riemann problem, and then
   * compute and return the Riemann data [rho, u, p, a] (used in the
   * approximative Riemann solver).
   */
  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline std::array<Number, 4>
  RiemannSolver<dim, Number>::riemann_data_from_state(
      const ProblemDescription &problem_description,
      const ProblemDescription::state_type<dim, Number> &U,
      const dealii::Tensor<1, dim, Number> &n_ij)
  {
    const auto rho = problem_description.density(U);
    const auto rho_inverse = Number(1.0) / rho;

    const auto m = problem_description.momentum(U);
    const auto proj_m = n_ij * m;
    const auto perp = m - proj_m * n_ij;

    const auto E = problem_description.total_energy(U) -
                   Number(0.5) * perp.norm_square() * rho_inverse;

    const auto state =
        ProblemDescription::state_type<1, Number>({rho, proj_m, E});
    const auto p = problem_description.pressure(state);
    const auto a = problem_description.speed_of_sound(state);

    return {{rho, proj_m * rho_inverse, p, a}};
  }


  template <int dim, typename Number>
#ifdef OBSESSIVE_INLINING
  DEAL_II_ALWAYS_INLINE inline
#endif
      std::tuple<Number, Number, unsigned int>
      RiemannSolver<dim, Number>::compute(
          const std::array<Number, 4> &riemann_data_i,
          const std::array<Number, 4> &riemann_data_j)
  {
    /*
     * Step 1:
     *
     * In case we iterate (in the Newton method) we need a good upper and
     * lower bound, p_1 < p_star < p_2, for finding phi(p_star) == 0. In case
     * we do not iterate (because the iteration is really expensive...) we
     * will need p_2 as an approximation to p_star.
     *
     * In any case we have to ensure that phi(p_2) >= 0 (and phi(p_1) <= 0).
     *
     * We will use three candidates, p_min, p_max and the two rarefaction
     * approximation p_star_tilde. We have (up to round-off errors) that
     * phi(p_star_tilde) >= 0. So this is a safe upper bound.
     *
     * Depending on the sign of phi(p_max) we select the following ranges:
     *
     * phi(p_max) <  0:
     *   p_1  <-  p_max   and   p_2  <-  p_star_tilde
     *
     * phi(p_max) >= 0:
     *   p_1  <-  p_min   and   p_2  <-  min(p_max, p_star_tilde)
     *
     * Nota bene:
     *
     *  - The special case phi(p_max) == 0 as discussed in [1] is already
     *    contained in the second condition.
     *
     *  - In principle, we would have to treat the case phi(p_min) > 0 as
     *    well. This corresponds to two expansion waves and a good estimate
     *    for the wavespeed is obtained by setting p_star = 0 and computing
     *    lambda_max with that.
     *    However, it turns out that numerically in this case the
     *    two-rarefaction approximation p_star_tilde is already an
     *    excellent guess and we will have
     *
     *      0 < p_star <= p_star_tilde <= p_min <= p_max.
     *
     *    So let's simply detect this case numerically by checking for p_2 <
     *    p_1 and setting p_1 <- 0 if necessary.
     */

    const Number p_min = std::min(riemann_data_i[2], riemann_data_j[2]);
    const Number p_max = std::max(riemann_data_i[2], riemann_data_j[2]);

    const Number p_star_tilde =
        p_star_two_rarefaction(riemann_data_i, riemann_data_j);

    const Number phi_p_max = phi_of_p_max(riemann_data_i, riemann_data_j);

    Number p_2 =
        dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
            phi_p_max, Number(0.), p_star_tilde, std::min(p_max, p_star_tilde));

    /* If we do no Newton iteration, cut it short: */

    if constexpr (newton_max_iter_ == 0) {
      const Number lambda_max =
          compute_lambda(riemann_data_i, riemann_data_j, p_2);
      return {lambda_max, p_2, -1};
    }

    Number p_1 =
        dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
            phi_p_max, Number(0.), p_max, p_min);

    /*
     * Ensure that p_1 < p_2. If we hit a case with two expansions we might
     * indeed have that p_star_tilde < p_1. Set p_1 = p_2 in this case.
     */

    p_1 = dealii::compare_and_apply_mask<
        dealii::SIMDComparison::less_than_or_equal>(p_1, p_2, p_1, p_2);

    /*
     * Step 2: Perform quadratic Newton iteration.
     *
     * See [1], p. 915f (4.8) and (4.9)
     */

    auto [gap, lambda_max] =
        compute_gap(riemann_data_i, riemann_data_j, p_1, p_2);

    unsigned int i = 0;
    for (; i < newton_max_iter_; ++i) {

      /* We return our current guess if we reach the tolerance... */
      if (std::max(Number(0.), gap - newton_eps<Number>) == Number(0.))
        break;

      // FIXME: Fuse these computations:
      const Number phi_p_1 = phi(riemann_data_i, riemann_data_j, p_1);
      const Number phi_p_2 = phi(riemann_data_i, riemann_data_j, p_2);
      const Number dphi_p_1 = dphi(riemann_data_i, riemann_data_j, p_1);
      const Number dphi_p_2 = dphi(riemann_data_i, riemann_data_j, p_2);

      quadratic_newton_step(p_1, p_2, phi_p_1, phi_p_2, dphi_p_1, dphi_p_2);

      /* Update  lambda_max and gap: */
      {
        auto [gap_new, lambda_max_new] =
            compute_gap(riemann_data_i, riemann_data_j, p_1, p_2);
        gap = gap_new;
        lambda_max = lambda_max_new;
      }
    }

#ifdef CHECK_BOUNDS
    const auto phi_p_star = phi(riemann_data_i, riemann_data_j, p_2);
    AssertThrowSIMD(
        phi_p_star,
        [](auto val) { return val >= -newton_eps<ScalarNumber>; },
        dealii::ExcMessage("Invalid state in Riemann problem."));
#endif

    return {lambda_max, p_2, i};
  }


  template <int dim, typename Number>
#ifdef OBSESSIVE_INLINING
  DEAL_II_ALWAYS_INLINE inline
#endif
      std::tuple<Number, Number, unsigned int>
      RiemannSolver<dim, Number>::compute(
          const state_type &U_i,
          const state_type &U_j,
          const dealii::Tensor<1, dim, Number> &n_ij)
  {
    const auto riemann_data_i =
        riemann_data_from_state(problem_description, U_i, n_ij);
    const auto riemann_data_j =
        riemann_data_from_state(problem_description, U_j, n_ij);

    return compute(riemann_data_i, riemann_data_j);
  }

} // namespace ryujin
