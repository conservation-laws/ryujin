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
      return (std::fabs(number) - number) / Number(2.0);
    }


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, and a
     * (normalized) "direction" n_ij, return the corresponding projected
     * state in the corresponding 1D Riemann problem.
     */
    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE dealii::Tensor<1, 3> projected_state(
        const typename ProblemDescription<dim, Number>::rank1_type U,
        const dealii::Tensor<1, dim, Number> &n_ij)
    {
      dealii::Tensor<1, 3> result;

      // rho:
      result[0] = U[0];

      // m:
      const auto m = ProblemDescription<dim, Number>::momentum(U);
      result[1] = n_ij * m;

      // E:
      const auto perpendicular_m = m - result[1] * n_ij;
      result[2] =
          U[1 + dim] - Number(0.5) * perpendicular_m.norm_square() / U[0];

      return result;
    }


    /**
     * For a projected state <code>projected_U</code> compute the
     * (physical) pressure p of the state.
     *
     * Recall that
     *   p = (gamma - 1.0)*rho*e = (gamma - 1.0)*(E - 0.5*m^2/rho)
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE Number pressure_from_projected_state(
        const Number gamma, const dealii::Tensor<1, 3> &projected_U)
    {
      return (gamma - Number(1.0)) *
             (projected_U[2] -
              Number(0.5) * projected_U[1] * projected_U[1] / projected_U[0]);
    }


    /**
     * For a projected state <code>projected_U</code> compute the
     * (physical) speed of sound.
     *
     * Recall that
     *   c^2 = gamma * p / rho / (1 - b * rho)
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE Number
    speed_of_sound_from_projected_state(const Number gamma,
                                        const Number b,
                                        const dealii::Tensor<1, 3> &projected_U)
    {
      const Number rho = projected_U[0];
      const Number p = pressure_from_projected_state(gamma, projected_U);

      return std::sqrt(gamma * p / rho / (Number(1.0) - b * rho));
    }


    /**
     * For a given projected state <code>projected_U</code> compute the
     * Riemann data [rho_Z, u_Z, p_Z, a_Z, A_Z, B_Z] (used in the
     * approximative Riemman solver).
     *
     * FIXME: Describe state in more detail.
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE std::array<Number, 6>
    riemann_data_from_projected_state(const Number gamma,
                                      const Number b,
                                      const dealii::Tensor<1, 3> &projected_U)
    {
      std::array<Number, 6> result;

      // rho_Z:
      result[0] = projected_U[0];
      // u_Z:
      result[1] = projected_U[1] / projected_U[0];
      // p_Z:
      result[2] = pressure_from_projected_state(gamma, projected_U);
      // a_Z:
      result[3] = speed_of_sound_from_projected_state(gamma, b, projected_U);
      // A_Z:
      result[4] = Number(2.0) * (Number(1.0) - b * projected_U[0]) /
                  (gamma + Number(1.0)) / projected_U[0];
      // B_Z:
      result[5] = (gamma - Number(1.0)) / (gamma + Number(1.0)) * result[2];

      return result;
    }


    /**
     * FIXME: Write a lengthy explanation.
     *
     * See [1], page 912, (3.4).
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE Number
    f_Z(const Number gamma,
        const Number b,
        const std::array<Number, 6> &primitive_state,
        const Number &p)
    {
      const auto &[rho_Z, u_Z, p_Z, a_Z, A_Z, B_Z] = primitive_state;

      if (p >= p_Z) {
        return (p - p_Z) * std::sqrt(A_Z / (p + B_Z));

      } else {

        const Number tmp =
            std::pow(p / p_Z, (gamma - Number(1.0)) / Number(2.0) / gamma) -
            Number(1.0);
        return Number(2.0) * a_Z * (Number(1.0) - b * rho_Z) /
               (gamma - Number(1.0)) * tmp;
      }
    }


    /**
     * FIXME: Write a lengthy explanation.
     *
     * See [1], page 912, (3.4). FIXME find equation defining the
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE Number
    df_Z(const Number gamma,
         const Number b,
         const std::array<Number, 6> &primitive_state,
         const Number &p)
    {
      const auto &[rho_Z, u_Z, p_Z, a_Z, A_Z, B_Z] = primitive_state;

      if (p >= p_Z) {
        /* Derivative of (p - p_Z) * std::sqrt(A_Z / (p + B_Z)): */
        return std::sqrt(A_Z / (p + B_Z)) *
               (Number(1.0) - Number(0.5) * (p - p_Z) / (p + B_Z));

      } else {

        /* Derivative of std::pow(p / p_Z, (gamma - 1.) / 2. / gamma) - 1.*/
        const Number tmp =
            (gamma - Number(1.0)) / Number(2.) / gamma *
            std::pow(p / p_Z, (Number(-1.0) - gamma) / Number(2.0) / gamma) /
            p_Z;
        return Number(2.) * a_Z * (Number(1.0) - b * rho_Z) /
               (gamma - Number(1.0)) * tmp;
      }
    }


    /**
     * FIXME: Write a lengthy explanation.
     *
     * See [1], page 912, (3.3).
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE Number
    phi(const Number gamma,
        const Number b,
        const std::array<Number, 6> &riemann_data_i,
        const std::array<Number, 6> &riemann_data_j,
        const Number &p)
    {
      const Number &u_i = riemann_data_i[1];
      const Number &u_j = riemann_data_j[1];

      return f_Z(gamma, b, riemann_data_i, p) +
             f_Z(gamma, b, riemann_data_j, p) + u_j - u_i;
    }


    /**
     * FIXME: Write a lengthy explanation.
     *
     * See [1], page 912, (3.3). FIXME find equation defining the
     * derivative.
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE Number
    dphi(const Number gamma,
         const Number b,
         const std::array<Number, 6> &riemann_data_i,
         const std::array<Number, 6> &riemann_data_j,
         const Number &p)
    {
      return df_Z(gamma, b, riemann_data_i, p) +
             df_Z(gamma, b, riemann_data_j, p);
    }


    /**
     * see [1], page 912, (3.7)
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE Number
    lambda1_minus(const Number gamma,
                  const std::array<Number, 6> &riemann_data,
                  const Number p_star)
    {
      const auto &[rho_Z, u_Z, p_Z, a_Z, A_Z, B_Z] = riemann_data;

      const Number factor = (gamma + Number(1.0)) / Number(2.0) / gamma;
      const Number tmp = positive_part((p_star - p_Z) / p_Z);
      return u_Z - a_Z * std::sqrt(Number(1.0) + factor * tmp);
    }


    /**
     * see [1], page 912, (3.8)
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE Number
    lambda3_plus(const Number gamma,
                 const std::array<Number, 6> &primitive_state,
                 const Number p_star)
    {
      const auto &[rho_Z, u_Z, p_Z, a_Z, A_Z, B_Z] = primitive_state;

      const Number factor = (gamma + Number(1.0)) / Number(2.0) / gamma;
      const Number tmp = positive_part((p_star - p_Z) / p_Z);
      return u_Z + a_Z * std::sqrt(Number(1.0) + factor * tmp);
    }


    /**
     * Two-rarefaction approximation to p_star computed for two primitive
     * states <code>riemann_data_i</code> and
     * <code>riemann_data_j</code>. See [1], page 914, (4.3)
     */
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE Number
    p_star_two_rarefaction(const Number gamma,
                           const Number b,
                           const std::array<Number, 6> &riemann_data_i,
                           const std::array<Number, 6> &riemann_data_j)
    {
      const auto &[rho_i, u_i, p_i, a_i, A_i, B_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, a_j, A_j, B_j] = riemann_data_j;

      /*
       * Notar bene (cf. [1, (4.3)]):
       *   a_Z^0 * sqrt(1 - b * rho_Z) = a_Z * (1 - b * rho_Z)
       * We have computed a_Z already, so we are simply going to use this
       * identity below:
       */

      const Number tmp_i = Number(1.0) - b * rho_i;
      const Number tmp_j = Number(1.0) - b * rho_j;

      const Number numerator = a_i * tmp_i + a_j * tmp_j -
                               (gamma - Number(1.)) / Number(2.0) * (u_j - u_i);

      const Number denominator =
          a_i * tmp_i *
              std::pow(p_i / p_j,
                       -Number(1.0) * (gamma - Number(1.)) / Number(2.) /
                           gamma) +
          a_j * tmp_j * Number(1.0);

      return p_j * std::pow(numerator / denominator,
                            Number(2.0) * gamma / (gamma - Number(1.0)));
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
    compute_gap(const Number gamma,
                const std::array<Number, 6> &riemann_data_i,
                const std::array<Number, 6> &riemann_data_j,
                const Number p_1,
                const Number p_2)
    {
      const Number nu_11 = lambda1_minus(gamma, riemann_data_i, p_2 /*SIC!*/);
      const Number nu_12 = lambda1_minus(gamma, riemann_data_i, p_1 /*SIC!*/);

      const Number nu_31 = lambda3_plus(gamma, riemann_data_j, p_1);
      const Number nu_32 = lambda3_plus(gamma, riemann_data_j, p_2);

      const Number lambda_max =
          std::max(positive_part(nu_32), negative_part(nu_11));

      const Number gap =
          std::max(std::abs(nu_32 - nu_31), std::abs(nu_12 - nu_11));

      return {gap, lambda_max};
    }

  } /*anonymous namespace*/


  template <int dim, typename Number>
  std::tuple<Number, Number, unsigned int> RiemannSolver<dim, Number>::compute(
      const rank1_type U_i,
      const rank1_type U_j,
      const dealii::Tensor<1, dim, Number> &n_ij)
  {
    constexpr Number gamma = ProblemDescription<dim, Number>::gamma;
    constexpr Number b = ProblemDescription<dim, Number>::b;

    /*
     * Step 1: Compute projected 1D states.
     */

    const auto projected_U_i = projected_state(U_i, n_ij);
    const auto projected_U_j = projected_state(U_j, n_ij);

    const auto riemann_data_i =
        riemann_data_from_projected_state(gamma, b, projected_U_i);
    const auto riemann_data_j =
        riemann_data_from_projected_state(gamma, b, projected_U_j);

    return compute(riemann_data_i, riemann_data_j);
  }


  template <int dim, typename Number>
  template <unsigned int max_iter>
  std::tuple<Number, Number, unsigned int> RiemannSolver<dim, Number>::compute(
      const std::array<Number, 6> &riemann_data_i,
      const std::array<Number, 6> &riemann_data_j)
  {
    constexpr Number gamma = ProblemDescription<dim, Number>::gamma;
    constexpr Number b = ProblemDescription<dim, Number>::b;

    const Number p_min = std::min(riemann_data_i[2], riemann_data_j[2]);
    const Number p_max = std::max(riemann_data_i[2], riemann_data_j[2]);

    /*
     * Step 2: Shortcuts.
     *
     * In a number of cases we actually do not need to do a Newton search
     * for the optimal lambda upper bound, but know the answer right
     * away.
     */

    const Number phi_p_min =
        phi(gamma, b, riemann_data_i, riemann_data_j, p_min);

    if (phi_p_min >= 0.) {
      const Number p_star = 0.;
      const Number lambda1 = lambda1_minus(gamma, riemann_data_i, p_star);
      const Number lambda3 = lambda3_plus(gamma, riemann_data_j, p_star);
      const Number lambda_max = std::max(std::abs(lambda1), std::abs(lambda3));
      return {lambda_max, p_star, 0};
    }

    const Number phi_p_max =
        phi(gamma, b, riemann_data_i, riemann_data_j, p_max);

    if (std::abs(phi_p_max) <= newton_eps_) {
      const Number p_star = p_max;
      const Number lambda1 = lambda1_minus(gamma, riemann_data_i, p_star);
      const Number lambda3 = lambda3_plus(gamma, riemann_data_j, p_star);
      const Number lambda_max = std::max(std::abs(lambda1), std::abs(lambda3));
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
        p_star_two_rarefaction(gamma, b, riemann_data_i, riemann_data_j);

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
          compute_gap(gamma, riemann_data_i, riemann_data_j, p_1, p_2);

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
      if (i + 1 >= max_iter)
        return {lambda_max, p_2, std::numeric_limits<unsigned int>::max()};

#if DEBUG
      {
        const Number phi_p_1 =
            phi(gamma, b, riemann_data_i, riemann_data_j, p_1);
        const Number phi_p_2 =
            phi(gamma, b, riemann_data_i, riemann_data_j, p_2);
        Assert(phi_p_1 <= 0. && phi_p_2 >= 0.,
               dealii::ExcMessage("Houston, we have a problem!"));
      }
#endif

      /*
       * This is expensive:
       */

      const Number phi_p_1 = phi(gamma, b, riemann_data_i, riemann_data_j, p_1);
      const Number dphi_p_1 =
          dphi(gamma, b, riemann_data_i, riemann_data_j, p_1);
      const Number phi_p_2 = phi(gamma, b, riemann_data_i, riemann_data_j, p_2);
      const Number dphi_p_2 =
          dphi(gamma, b, riemann_data_i, riemann_data_j, p_2);

      /*
       * Sanity checks:
       *  * phi is monotone increasing and concave down: the derivative
       *    has to be positive, both function values have to be different
       *  * p_1 < p_2
       */

      Assert(dphi_p_1 > 0., dealii::ExcMessage("Houston, we have a problem!"));
      Assert(dphi_p_2 > 0., dealii::ExcMessage("Houston, we have a problem!"));
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
            gamma, riemann_data_i, riemann_data_j, p_2 /*SIC!*/, p_1 /*SIC!*/);
        return {lambda_max, p_2, i + 1};
      }

    } while (i++ < max_iter);

    __builtin_unreachable();
  }


} /* namespace grendel */

#endif /* RIEMANN_SOLVER_TEMPLATE_H */
