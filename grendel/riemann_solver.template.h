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
     * This combines the calculation of phi both against p_min and p_max in a
     * single call to reduce the number of calls to pow(). It is inlining the
     * content of f() and choosing only the relevant paths.
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE std::array<Number, 2>
    phi_twosided(const std::array<Number, 4> &riemann_data_i,
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

      const Number p_min = std::min(p_i, p_j);
      const Number p_max = std::max(p_i, p_j);

      // The exponent path is only selected for the branch where we compute
      // p_min / p_max, in the other three cases the radicand case is selected
      const auto exponent =
          (gamma - ScalarNumber(1.)) * ScalarNumber(0.5) * gamma_inverse;
      const Number factor = grendel::pow(p_min / p_max, exponent) - Number(1.);
      const Number false_value_i =
          factor * ScalarNumber(2.) * a_i * gamma_minus_one_inverse;
      const Number false_value_j =
          factor * ScalarNumber(2.) * a_j * gamma_minus_one_inverse;

      const Number radicand_inverse_i_1 = ScalarNumber(0.5) * rho_i *
                                          ((gamma + ScalarNumber(1.)) * p_min +
                                           (gamma - ScalarNumber(1.)) * p_i);
      const Number true_value_i_1 =
          (p_min - p_i) / std::sqrt(radicand_inverse_i_1);
      const Number radicand_inverse_i_2 = ScalarNumber(0.5) * rho_i *
                                          ((gamma + ScalarNumber(1.)) * p_max +
                                           (gamma - ScalarNumber(1.)) * p_i);
      const Number true_value_i_2 =
          (p_max - p_i) / std::sqrt(radicand_inverse_i_2);
      const Number radicand_inverse_j_1 = ScalarNumber(0.5) * rho_j *
                                          ((gamma + ScalarNumber(1.)) * p_min +
                                           (gamma - ScalarNumber(1.)) * p_j);
      const Number true_value_j_1 =
          (p_min - p_j) / std::sqrt(radicand_inverse_j_1);
      const Number radicand_inverse_j_2 = ScalarNumber(0.5) * rho_j *
                                          ((gamma + ScalarNumber(1.)) * p_max +
                                           (gamma - ScalarNumber(1.)) * p_j);
      const Number true_value_j_2 =
          (p_max - p_j) / std::sqrt(radicand_inverse_j_2);

      // The p_max part always selects the 'true' branch, whereas we need to
      // make a selection for the p_min part
      return {true_value_i_2 + true_value_j_2 + u_j - u_i,
              dealii::compare_and_apply_mask<
                  dealii::SIMDComparison::greater_than_or_equal>(
                  p_min, p_i, true_value_i_1, false_value_i) +
                  dealii::compare_and_apply_mask<
                      dealii::SIMDComparison::greater_than_or_equal>(
                      p_min, p_j, true_value_j_1, false_value_j) +
                  u_j - u_i};
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
    if constexpr (newton_max_iter_ == 0) {

      /*
       * SIMDified version of the two-rarefaction approximation.
       */

      // const Number p_min = std::min(riemann_data_i[2], riemann_data_j[2]);
      const Number p_max = std::max(riemann_data_i[2], riemann_data_j[2]);

      const auto &[phi_p_max, phi_p_min] =
          phi_twosided(riemann_data_i, riemann_data_j);

      const Number p_star_tilde =
          p_star_two_rarefaction(riemann_data_i, riemann_data_j);

      Number p_star =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              phi_p_max,
              Number(0.),
              p_star_tilde,
              std::min(p_max, p_star_tilde));

      p_star = dealii::compare_and_apply_mask<
          dealii::SIMDComparison::less_than_or_equal>(
          std::abs(phi_p_max), Number(newton_eps_), p_max, p_star);

      p_star = dealii::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          phi_p_min, Number(0.), Number(0.), p_star);

      const Number lambda_max =
          compute_lambda(riemann_data_i, riemann_data_j, p_star);
      return {lambda_max, p_star, -1};

    } else {

      /*
       * Full approximate Riemann solver, currently only implemented for
       * nonSIMD number types
       */

      static_assert(std::is_same<Number, double>::value ||
                        std::is_same<Number, float>::value,
                    "Currently not ported to SIMD");

      const Number p_min = std::min(riemann_data_i[2], riemann_data_j[2]);
      const Number p_max = std::max(riemann_data_i[2], riemann_data_j[2]);

      const Number phi_p_min = phi(riemann_data_i, riemann_data_j, p_min);

      if (phi_p_min >= 0.) {
        const Number p_star = 0.;
        const Number lambda_max =
            compute_lambda(riemann_data_i, riemann_data_j, p_star);
        return {lambda_max, p_star, 0};
      }

      const Number phi_p_max = phi(riemann_data_i, riemann_data_j, p_max);

      if (std::abs(phi_p_max) <= newton_eps_) {
        const Number p_star = p_max;
        const Number lambda_max =
            compute_lambda(riemann_data_i, riemann_data_j, p_star);
        return {lambda_max, p_star, 0};
      }

      /*
       * Step 3: Prepare quadratic Newton method.
       *
       * We need a good upper and lower bound, p_1 < p_star < p_2, for the
       * Newton method. (Ideally, for a moderate tolerance we might not
       * iterate at all.)
       */

      const Number p_star_tilde =
          p_star_two_rarefaction(riemann_data_i, riemann_data_j);

      Number p_1 = (phi_p_max < 0.) ? p_max : p_min;
      Number p_2 =
          (phi_p_max < 0.) ? p_star_tilde : std::min(p_max, p_star_tilde);

      /*
       * Step 4: Perform quadratic Newton iteration.
       *
       * See [1], p. 915f (4.8) and (4.9)
       */

      unsigned int i = 0;
      do {
        const auto [gap, lambda_max] =
            compute_gap(riemann_data_i, riemann_data_j, p_1, p_2);

        /*
         * We return our current guess if we either reach the tolerance...
         */
        if (gap < newton_eps_)
          return {lambda_max, p_2, i};

        /*
         * ... or if we reached the number of allowed Newton iterations.
         * lambda_max is a guaranteed upper bound, in the worst case we
         * overestimated the result.
         */
        if (i + 1 >= newton_max_iter_)
          return {lambda_max, p_2, std::numeric_limits<unsigned int>::max()};

#if DEBUG
        {
          const Number phi_p_1 = phi(riemann_data_i, riemann_data_j, p_1);
          const Number phi_p_2 = phi(riemann_data_i, riemann_data_j, p_2);
          Assert(phi_p_1 <= 0. && phi_p_2 >= 0.,
                 dealii::ExcMessage("Houston, we have a problem!"));
        }
#endif

        /*
         * This is expensive:
         */

        const Number phi_p_1 = phi(riemann_data_i, riemann_data_j, p_1);
        const Number dphi_p_1 = dphi(riemann_data_i, riemann_data_j, p_1);
        const Number phi_p_2 = phi(riemann_data_i, riemann_data_j, p_2);
        const Number dphi_p_2 = dphi(riemann_data_i, riemann_data_j, p_2);

        /*
         * Sanity checks:
         *  * phi is monotone increasing and concave down: the derivative
         *    has to be positive, both function values have to be different
         *  * p_1 < p_2
         */

        Assert(dphi_p_1 > 0.,
               dealii::ExcMessage("Houston, we have a problem!"));
        Assert(dphi_p_2 > 0.,
               dealii::ExcMessage("Houston, we have a problem!"));
        Assert(phi_p_1 < phi_p_2,
               dealii::ExcMessage("Houston, we have a problem!"));
        Assert(p_1 < p_2, dealii::ExcMessage("Houston, we have a problem!"));

        /*
         * Compute divided differences
         */

        const Number dd_11 = dphi_p_1;
        const Number dd_12 = (phi_p_2 - phi_p_1) / (p_2 - p_1);
        const Number dd_22 = dphi_p_2;

        const Number dd_112 = (dd_12 - dd_11) / (p_2 - p_1);
        const Number dd_122 = (dd_22 - dd_12) / (p_2 - p_1);

        /* Update left point: */
        const Number discriminant_1 =
            dphi_p_1 * dphi_p_1 - Number(4.0) * phi_p_1 * dd_112;
        Assert(discriminant_1 > 0.0,
               dealii::ExcMessage("Houston, we have a problem!"));
        if (discriminant_1 > 0.)
          p_1 = p_1 -
                Number(2.0) * phi_p_1 / (dphi_p_1 + std::sqrt(discriminant_1));

        /* Update right point: */
        const Number discriminant_2 =
            dphi_p_2 * dphi_p_2 - Number(4.0) * phi_p_2 * dd_122;
        Assert(discriminant_2 > 0.0,
               dealii::ExcMessage("Houston, we have a problem!"));
        if (discriminant_2 > 0.)
          p_2 = p_2 -
                Number(2.0) * phi_p_2 / (dphi_p_2 + std::sqrt(discriminant_2));


        /* We have found our root (up to roundoff erros): */
        if (p_1 >= p_2) {
          /*
           * p_1 has changed position with p_2, it is now on the right side,
           * so call the compute_gap function with reversed parameters:
           */
          const auto [gap, lambda_max] = compute_gap(
              riemann_data_i, riemann_data_j, p_2 /*SIC!*/, p_1 /*SIC!*/);
          return {lambda_max, p_2, i + 1};
        }

      } while (i++ < newton_max_iter_);

      __builtin_unreachable();
    } /* if constexpr */
  }

} /* namespace grendel */

#endif /* RIEMANN_SOLVER_TEMPLATE_H */
