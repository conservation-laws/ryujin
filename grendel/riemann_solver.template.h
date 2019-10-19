#ifndef RIEMANN_SOLVER_TEMPLATE_H
#define RIEMANN_SOLVER_TEMPLATE_H

#include "riemann_solver.h"

namespace grendel
{
  using namespace dealii;


  /*
   * HERE BE DRAGONS!
   */


  namespace
  {
    /**
     * Return the positive part of a number.
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE Number positive_part(const Number number)
    {
      return (std::abs(number) + number) / Number(2.0);
    }


    /**
     * Return the negative part of a number.
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE Number negative_part(const Number number)
    {
      return (std::abs(number) - number) / Number(2.0);
    }


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, and a
     * (normalized) "direction" n_ij, first compute the corresponding
     * projected state in the corresponding 1D Riemann problem, and then
     * compute and return the Riemann data [rho, u, p, a] (used in the
     * approximative Riemman solver).
     */
    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE std::array<Number, 4> riemann_data_from_state(
        const typename ProblemDescription<dim, Number>::rank1_type U,
        const dealii::Tensor<1, dim, Number> &n_ij)
    {
      typename ProblemDescription<1, Number>::rank1_type projected;

      projected[0] = U[0];

      const Number inv_density = Number(1.0) / U[0];

      const auto m = ProblemDescription<dim, Number>::momentum(U);
      projected[1] = n_ij * m;

      const auto perp = m - projected[1] * n_ij;
      projected[2] =
          U[1 + dim] - Number(0.5) * perp.norm_square() * inv_density;

      std::array<Number, 4> result;

      result[0] = projected[0];               // rho
      result[1] = projected[1] * inv_density; // u
      result[2] =
          ProblemDescription<1, Number>::pressure(projected, inv_density);
      result[3] =
          ProblemDescription<1, Number>::speed_of_sound(projected, inv_density);

      return result;
    }


    /**
     * FIXME: Write a lengthy explanation.
     *
     * See [1], page 912, (3.4).
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE Number
    f(const std::array<Number, 4> &primitive_state, const Number &p_star)
    {
      using ScalarNumber = typename get_value_type<Number>::type;

      constexpr auto gamma = ProblemDescription<1, Number>::gamma;
      constexpr auto gamma_inverse =
          ProblemDescription<1, Number>::gamma_inverse;
      constexpr auto gamma_minus_one_inverse =
          ProblemDescription<1, Number>::gamma_minus_one_inverse;
      const auto &[rho, u, p, a] = primitive_state;

      const Number radicand_inverse = ScalarNumber(0.5) * rho *
                                      ((gamma + ScalarNumber(1.)) * p_star +
                                       (gamma - ScalarNumber(1.)) * p);
      const Number true_value = (p_star - p) / std::sqrt(radicand_inverse);

      const auto exponent =
          (gamma - ScalarNumber(1.)) * ScalarNumber(0.5) * gamma_inverse;
      const Number factor = grendel::pow(p_star / p, exponent) - Number(1.);
      const auto false_value =
          factor * ScalarNumber(2.) * a * gamma_minus_one_inverse;

      return dealii::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_star, p, true_value, false_value);
    }

    /**
     * FIXME: Write a lengthy explanation.
     *
     * See [1], page 912, (3.4).
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE Number
    df(const std::array<Number, 4> &primitive_state, const Number &p_star)
    {
      using ScalarNumber = typename get_value_type<Number>::type;

      constexpr auto gamma = ProblemDescription<1, Number>::gamma;
      constexpr auto gamma_inverse =
          ProblemDescription<1, Number>::gamma_inverse;
      constexpr auto gamma_minus_one_inverse =
          ProblemDescription<1, Number>::gamma_minus_one_inverse;
      constexpr auto gamma_plus_one_inverse =
          ProblemDescription<1, Number>::gamma_plus_one_inverse;
      const auto &[rho, u, p, a] = primitive_state;

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
                            gamma_inverse * grendel::pow(p_star / p, exponent) /
                            p;
      const auto false_value =
          factor * ScalarNumber(2.) * a * gamma_minus_one_inverse;

      return dealii::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_star, p, true_value, false_value);
    }


    /**
     * FIXME: Write a lengthy explanation.
     *
     * See [1], page 912, (3.3).
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE Number
    phi(const std::array<Number, 4> &riemann_data_i,
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
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE Number
    phi_of_p_max(const std::array<Number, 4> &riemann_data_i,
                 const std::array<Number, 4> &riemann_data_j)
    {
      using ScalarNumber = typename get_value_type<Number>::type;

      constexpr auto gamma = ProblemDescription<1, Number>::gamma;

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


    /**
     * FIXME: Write a lengthy explanation.
     *
     * See [1], page 912, (3.3).
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE Number
    dphi(const std::array<Number, 4> &riemann_data_i,
         const std::array<Number, 4> &riemann_data_j,
         const Number &p)
    {
      return df(riemann_data_i, p) + df(riemann_data_j, p);
    }


    /**
     * see [1], page 912, (3.7)
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE Number lambda1_minus(
        const std::array<Number, 4> &riemann_data, const Number p_star)
    {
      using ScalarNumber = typename get_value_type<Number>::type;

      constexpr auto gamma = ProblemDescription<1, Number>::gamma;
      constexpr auto gamma_inverse =
          ProblemDescription<1, Number>::gamma_inverse;
      const auto &[rho, u, p, a] = riemann_data;

      const auto factor =
          (gamma + ScalarNumber(1.0)) * ScalarNumber(0.5) * gamma_inverse;
      const Number tmp = positive_part((p_star - p) / p);

      return u - a * std::sqrt(Number(1.0) + factor * tmp);
    }


    /**
     * see [1], page 912, (3.8)
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE Number lambda3_plus(
        const std::array<Number, 4> &primitive_state, const Number p_star)
    {
      using ScalarNumber = typename get_value_type<Number>::type;

      constexpr auto gamma = ProblemDescription<1, Number>::gamma;
      constexpr auto gamma_inverse =
          ProblemDescription<1, Number>::gamma_inverse;
      const auto &[rho, u, p, a] = primitive_state;

      const Number factor =
          (gamma + ScalarNumber(1.0)) * ScalarNumber(0.5) * gamma_inverse;
      const Number tmp = positive_part((p_star - p) / p);
      return u + a * std::sqrt(Number(1.0) + factor * tmp);
    }


    /**
     * Two-rarefaction approximation to p_star computed for two primitive
     * states <code>riemann_data_i</code> and
     * <code>riemann_data_j</code>. See [1], page 914, (4.3)
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE Number
    p_star_two_rarefaction(const std::array<Number, 4> &riemann_data_i,
                           const std::array<Number, 4> &riemann_data_j)
    {
      using ScalarNumber = typename get_value_type<Number>::type;

      constexpr auto gamma = ProblemDescription<1, Number>::gamma;
      constexpr auto gamma_inverse =
          ProblemDescription<1, Number>::gamma_inverse;
      constexpr auto gamma_minus_one_inverse =
          ProblemDescription<1, Number>::gamma_minus_one_inverse;
      const auto &[rho_i, u_i, p_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, a_j] = riemann_data_j;

      /*
       * Notar bene (cf. [1, (4.3)]):
       *   a_Z^0 * sqrt(1 - b * rho_Z) = a_Z * (1 - b * rho_Z)
       * We have computed a_Z already, so we are simply going to use this
       * identity below:
       */

      const auto factor = (gamma - ScalarNumber(1.)) * ScalarNumber(0.5);

      const Number numerator = a_i + a_j - factor * (u_j - u_i);

      const Number denominator =
          a_i * grendel::pow(p_i / p_j, -factor * gamma_inverse) + a_j;

      const auto exponent = ScalarNumber(2.0) * gamma * gamma_minus_one_inverse;

      return p_j * grendel::pow(numerator / denominator, exponent);
    }


    /**
     * For two given primitive states <code>riemann_data_i</code> and
     * <code>riemann_data_j</code>, and two guesses p_1 < p_2, compute
     * the gap in lambda between both guesses.
     *
     * See [1], page 914, (4.4a), (4.4b), (4.5), and (4.6)
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE std::array<Number, 2>
    compute_gap(const std::array<Number, 4> &riemann_data_i,
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

      return {gap, lambda_max};
    }


    /**
     * For two given primitive states <code>riemann_data_i</code> and
     * <code>riemann_data_j</code>, and a guess p_2, compute an upper bound
     * for lambda.
     *
     * This is the same lambda_max as computed by compute_gap.
     *
     * See [1], page 914, (4.4a), (4.4b), (4.5), and (4.6)
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE Number
    compute_lambda(const std::array<Number, 4> &riemann_data_i,
                   const std::array<Number, 4> &riemann_data_j,
                   const Number p_star)
    {
      const Number nu_11 = lambda1_minus(riemann_data_i, p_star);
      const Number nu_32 = lambda3_plus(riemann_data_j, p_star);

      return std::max(positive_part(nu_32), negative_part(nu_11));
    }

  } /*anonymous namespace*/


  template <int dim, typename Number>
  std::tuple<Number, Number, unsigned int> RiemannSolver<dim, Number>::compute(
      const rank1_type U_i,
      const rank1_type U_j,
      const dealii::Tensor<1, dim, Number> &n_ij)
  {
    const auto riemann_data_i = riemann_data_from_state(U_i, n_ij);
    const auto riemann_data_j = riemann_data_from_state(U_j, n_ij);

    return compute(riemann_data_i, riemann_data_j);
  }


  template <int dim, typename Number>
  std::tuple<Number, Number, unsigned int> RiemannSolver<dim, Number>::compute(
      const std::array<Number, 4> &riemann_data_i,
      const std::array<Number, 4> &riemann_data_j)
  {
    /*
     * Step 1:
     *
     * In case we iterate (in the Newton method) we need a good upper and
     * lower bound, p_1 < p_star < p_2, for finding phi(p_star) == 0.
     *
     * In case we do not iterate (because the iteration iw really
     * expensive...) we will need p_2 as an approximation to p_star.
     *
     * In any case we have to ensure that phi(p_2) >= 0 (and phi(p_1) <=
     * 0).
     *
     * We will use three candidates, p_min, p_max and the two rarefaction
     * approximation p_star_tilde. We have (up to round-off errors) that
     * phi(p_star_tilde) >= 0. So this is a save upper bound.
     *
     * Depending on the sign of phi(p_max) we thus select the following
     * ranges:
     *
     * phi(p_max) <  0:
     *   p_1  <-  p_max   and   p_2  <-  p_star_tilde
     *
     * phi(p_max) >= 0:
     *   p_1  <-  p_min   and   p_2  <-  min(p_max, p_star_tilde)
     *
     * Notar bene:
     *
     *  - The case phi(p_max) == 0 as discussed as a special case
     *    in [1] is already contained in the second condition. We thus
     *    simply change the comparison to "phi(p_max) < -eps" to allow for
     *    numerical round-off errors.
     *
     *  - In principle, we would have to treat the case phi(p_min) > 0 as
     *    well. This corresponds to two expansion waves and a good estimate
     *    for the wavespeed is obtained by setting p_star = 0. and exiting.
     *    However, it turns out that numerically in this case the
     *    two-rarefaction approximation p_star_tilde is already an
     *    excellent guess (and we will set p_2 to this upper bound in both
     *    of the above two cases).
     *
     *    So let's happily take the risk by setting
     *      p_1 <- 0.   and  p_2  <-  p_star_tilde
     *    in this case.
     *
     *    Important: _NONE_ of these considerations changes the fact that
     *    the computed lambda_max is an upper bound on the maximum
     *    wavespeed. We might simply be a bit worse off in this case if it
     *    happens that p_star_tilde is a bad guess.
     */

    const Number p_min = std::min(riemann_data_i[2], riemann_data_j[2]);
    const Number p_max = std::max(riemann_data_i[2], riemann_data_j[2]);

    const Number p_star_tilde =
        p_star_two_rarefaction(riemann_data_i, riemann_data_j);

    const Number phi_p_max = phi_of_p_max(riemann_data_i, riemann_data_j);

    Number p_2 =
        dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
            phi_p_max,
            Number(-newton_eps_), /* prefer p_max if close to zero */
            p_star_tilde,
            std::min(p_max, p_star_tilde));

    if constexpr (newton_max_iter_ == 0) {

      /* If there is nothing to do, cut it short: */

      const Number lambda_max =
          compute_lambda(riemann_data_i, riemann_data_j, p_2);
      return {lambda_max, p_2, -1};
    }

    Number p_1 =
        dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
            phi_p_max,
            Number(newton_eps_), /* prefer p_max if close to zero */
            p_max,
            p_min);

    /*
     * Ensure that p_1 < p_2. In case we hit a case with two expansions we
     * might indeed have that p_star_tilde < p_1. Set p_1 = 0. in this case
     * (because this is the value we want to attain anyway).
     */

    p_1 = dealii::compare_and_apply_mask<
        dealii::SIMDComparison::less_than_or_equal>(p_1, p_2, p_1, Number(0.));

    /*
     * Step 2: Perform quadratic Newton iteration.
     *
     * See [1], p. 915f (4.8) and (4.9)
     */

    auto [gap, lambda_max] =
        compute_gap(riemann_data_i, riemann_data_j, p_1, p_2);

    unsigned int i = 0;
    for (; i < newton_max_iter_; ++i) {

      /*
       * We return our current guess if we reach the tolerance...
       */

      if (std::max(Number(0.), gap - Number(newton_eps_)) == Number(0.))
        break;

      // FIXME: Fuse these computations:
      const Number phi_p_1 = phi(riemann_data_i, riemann_data_j, p_1);
      const Number phi_p_2 = phi(riemann_data_i, riemann_data_j, p_2);
      const Number dphi_p_1 = dphi(riemann_data_i, riemann_data_j, p_1);
      const Number dphi_p_2 = dphi(riemann_data_i, riemann_data_j, p_2);

      /*
       * Compute divided differences
       */

      const Number dd_11 = dphi_p_1;
      const Number dd_12 = (phi_p_2 - phi_p_1) / (p_2 - p_1);
      const Number dd_22 = dphi_p_2;

      const Number dd_112 = (dd_12 - dd_11) / (p_2 - p_1);
      const Number dd_122 = (dd_22 - dd_12) / (p_2 - p_1);

      /* Update left and right point: */

      const auto discriminant_1 =
          std::abs(dphi_p_1 * dphi_p_1 - ScalarNumber(4.) * phi_p_1 * dd_112);
      const auto discriminant_2 =
          std::abs(dphi_p_2 * dphi_p_2 - ScalarNumber(4.) * phi_p_2 * dd_122);

      const auto denominator_1 = dphi_p_1 + std::sqrt(discriminant_1);
      const auto denominator_2 = dphi_p_2 + std::sqrt(discriminant_2);

      /* Make sure we do not produce NaNs: */

      auto t_1 =
          p_1 -
          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              std::abs(denominator_1),
              Number(newton_eps_),
              Number(0.),
              ScalarNumber(2.) * phi_p_1 / denominator_1);

      auto t_2 =
          p_2 -
          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              std::abs(denominator_2),
              Number(newton_eps_),
              Number(0.),
              ScalarNumber(2.) * phi_p_2 / denominator_2);

      /* Enforce bounds: */

      t_1 = std::max(p_1, t_1);
      t_2 = std::max(p_1, t_2);
      t_1 = std::min(p_2, t_1);
      t_2 = std::min(p_2, t_2);

      /* Ensure that always p_1 <= p_2: */

      p_1 = std::min(t_1, t_2);
      p_2 = std::max(t_1, t_2);

      /* Update  lambda_max and gap: */
      auto [gap_new, lambda_max_new] =
          compute_gap(riemann_data_i, riemann_data_j, p_1, p_2);
      gap = gap_new;
      lambda_max = lambda_max_new;
    }

    return {lambda_max, p_2, i};
  }

} /* namespace grendel */

#endif /* RIEMANN_SOLVER_TEMPLATE_H */
