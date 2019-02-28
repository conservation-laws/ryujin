#ifndef RIEMANN_SOLVER_TEMPLATE_H
#define RIEMANN_SOLVER_TEMPLATE_H

#include "riemann_solver.h"

namespace grendel
{
  using namespace dealii;

  template <int dim>
  RiemannSolver<dim>::RiemannSolver(
      const grendel::ProblemDescription<dim> &problem_description,
      const std::string &subsection)
      : ParameterAcceptor(subsection)
      , problem_description_(&problem_description)
  {
    eps_ = 1.e-10;
    add_parameter("newton eps", eps_, "Tolerance of the Newton secant solver");

    max_iter_ = 10;
    add_parameter("newton max iter",
                  max_iter_,
                  "Maximal number of iterations for the Newton secant solver");
  }


  /*
   * HERE BE DRAGONS!
   */


  namespace
  {
    /**
     * Return the positive part of a number.
     */
    inline DEAL_II_ALWAYS_INLINE double positive_part(const double number)
    {
      return (std::abs(number) + number) / 2.0;
    }


    /**
     * Return the negative part of a number.
     */
    inline DEAL_II_ALWAYS_INLINE double negative_part(const double number)
    {
      return (std::fabs(number) - number) / 2.0;
    }


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, and a
     * (normalized) "direction" n_ij, return the corresponding projected
     * state in the corresponding 1D Riemann problem.
     */
    template <int dim>
    inline DEAL_II_ALWAYS_INLINE dealii::Tensor<1, 3>
    projected_state(const typename ProblemDescription<dim>::rank1_type &U,
                    const dealii::Tensor<1, dim> &n_ij)
    {
      dealii::Tensor<1, 3> result;

      // rho:
      result[0] = U[0];

      // m:
      const auto m = ProblemDescription<dim>::momentum_vector(U);
      result[1] = n_ij * m;

      // E:
      const auto perpendicular_m = m - result[1] * n_ij;
      result[2] = U[1 + dim] - 0.5 * perpendicular_m.norm_square() / U[0];

      return result;
    }


    /**
     * For a projected state <code>projected_U</code> compute the
     * (physical) pressure p of the state.
     *
     * Recall that
     *   p = (gamma - 1.0)*rho*e = (gamma - 1.0)*(E - 0.5*m^2/rho)
     */
    inline DEAL_II_ALWAYS_INLINE double
    pressure_from_projected_state(const double gamma,
                                  const dealii::Tensor<1, 3> &projected_U)
    {
      return (gamma - 1.0) *
             (projected_U[2] -
              0.5 * projected_U[1] * projected_U[1] / projected_U[0]);
    }


    /**
     * For a projected state <code>projected_U</code> compute the
     * (physical) speed of sound.
     *
     * Recall that
     *   c^2 = gamma * p / rho / (1 - b * rho)
     */
    inline DEAL_II_ALWAYS_INLINE double
    speed_of_sound_from_projected_state(const double gamma,
                                        const double b,
                                        const dealii::Tensor<1, 3> &projected_U)
    {
      const double rho = projected_U[0];
      const double p = pressure_from_projected_state(gamma, projected_U);

      return std::sqrt(gamma * p / rho / (1.0 - b * rho));
    }


    /**
     * For a given projected state <code>projected_U</code> compute the
     * Riemann data [rho_Z, u_Z, p_Z, a_Z, A_Z, B_Z] (used in the
     * approximative Riemman solver).
     *
     * FIXME: Describe state in more detail.
     */
    inline DEAL_II_ALWAYS_INLINE std::array<double, 6>
    riemann_data_from_projected_state(const double gamma,
                                      const double b,
                                      const dealii::Tensor<1, 3> &projected_U)
    {
      std::array<double, 6> result;

      // rho_Z:
      result[0] = projected_U[0];
      // u_Z:
      result[1] = projected_U[1] / projected_U[0];
      // p_Z:
      result[2] = pressure_from_projected_state(gamma, projected_U);
      // a_Z:
      result[3] = speed_of_sound_from_projected_state(gamma, b, projected_U);
      // A_Z:
      result[4] =
          2.0 * (1.0 - b * projected_U[0]) / (gamma + 1.0) / projected_U[0];
      // B_Z:
      result[5] = (gamma - 1.0) / (gamma + 1.0) * result[2];

      return result;
    }


    /**
     * FIXME: Write a lengthy explanation.
     *
     * See [1], page 912, (3.4).
     */
    inline DEAL_II_ALWAYS_INLINE double
    f_Z(const double gamma,
        const double b,
        const std::array<double, 6> &primitive_state,
        const double &p)
    {
      const auto &[rho_Z, u_Z, p_Z, a_Z, A_Z, B_Z] = primitive_state;

      if (p >= p_Z) {
        return (p - p_Z) * std::sqrt(A_Z / (p + B_Z));

      } else {

        const double tmp = std::pow(p / p_Z, (gamma - 1.) / 2. / gamma) - 1.;
        return 2. * a_Z * (1. - b * rho_Z) / (gamma - 1.) * tmp;
      }
    }


    /**
     * FIXME: Write a lengthy explanation.
     *
     * See [1], page 912, (3.4). FIXME find equation defining the
     */
    inline DEAL_II_ALWAYS_INLINE double
    df_Z(const double gamma,
         const double b,
         const std::array<double, 6> &primitive_state,
         const double &p)
    {
      const auto &[rho_Z, u_Z, p_Z, a_Z, A_Z, B_Z] = primitive_state;

      if (p >= p_Z) {
        /* Derivative of (p - p_Z) * std::sqrt(A_Z / (p + B_Z)): */
        return std::sqrt(A_Z / (p + B_Z)) * (1. - 0.5 * (p - p_Z) / (p + B_Z));

      } else {

        /* Derivative of std::pow(p / p_Z, (gamma - 1.) / 2. / gamma) - 1.*/
        const double tmp = (gamma - 1.) / 2. / gamma *
                           std::pow(p / p_Z, (-1. - gamma) / 2. / gamma) / p_Z;
        return 2. * a_Z * (1. - b * rho_Z) / (gamma - 1.) * tmp;
      }
    }


    /**
     * FIXME: Write a lengthy explanation.
     *
     * See [1], page 912, (3.3).
     */
    inline DEAL_II_ALWAYS_INLINE double
    phi(const double gamma,
        const double b,
        const std::array<double, 6> &riemann_data_i,
        const std::array<double, 6> &riemann_data_j,
        const double &p)
    {
      const double &u_i = riemann_data_i[1];
      const double &u_j = riemann_data_j[1];

      return f_Z(gamma, b, riemann_data_i, p) +
             f_Z(gamma, b, riemann_data_j, p) + u_j - u_i;
    }


    /**
     * FIXME: Write a lengthy explanation.
     *
     * See [1], page 912, (3.3). FIXME find equation defining the
     * derivative.
     */
    inline DEAL_II_ALWAYS_INLINE double
    dphi(const double gamma,
         const double b,
         const std::array<double, 6> &riemann_data_i,
         const std::array<double, 6> &riemann_data_j,
         const double &p)
    {
      return df_Z(gamma, b, riemann_data_i, p) +
             df_Z(gamma, b, riemann_data_j, p);
    }


    /**
     * see [1], page 912, (3.7)
     */
    inline DEAL_II_ALWAYS_INLINE double
    lambda1_minus(const double gamma,
                  const std::array<double, 6> &riemann_data,
                  const double p_star)
    {
      const auto &[rho_Z, u_Z, p_Z, a_Z, A_Z, B_Z] = riemann_data;

      const double factor = (gamma + 1.0) / 2.0 / gamma;
      const double tmp = positive_part((p_star - p_Z) / p_Z);
      return u_Z - a_Z * std::sqrt(1.0 + factor * tmp);
    }


    /**
     * see [1], page 912, (3.8)
     */
    inline DEAL_II_ALWAYS_INLINE double
    lambda3_plus(const double gamma,
                 const std::array<double, 6> &primitive_state,
                 const double p_star)
    {
      const auto &[rho_Z, u_Z, p_Z, a_Z, A_Z, B_Z] = primitive_state;

      const double factor = (gamma + 1.0) / 2.0 / gamma;
      const double tmp = positive_part((p_star - p_Z) / p_Z);
      return u_Z + a_Z * std::sqrt(1.0 + factor * tmp);
    }


    /**
     * Two-rarefaction approximation to p_star computed for two primitive
     * states <code>riemann_data_i</code> and
     * <code>riemann_data_j</code>. See [1], page 914, (4.3)
     */
    inline DEAL_II_ALWAYS_INLINE double
    p_star_two_rarefaction(const double gamma,
                           const double b,
                           const std::array<double, 6> &riemann_data_i,
                           const std::array<double, 6> &riemann_data_j)
    {
      const auto &[rho_i, u_i, p_i, a_i, A_i, B_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, a_j, A_j, B_j] = riemann_data_j;

      /*
       * Notar bene (cf. [1, (4.3)]):
       *   a_Z^0 * sqrt(1 - b * rho_Z) = a_Z * (1 - b * rho_Z)
       * We have computed a_Z already, so we are simply going to use this
       * identity below:
       */

      const double tmp_i = 1. - b * rho_i;
      const double tmp_j = 1. - b * rho_j;

      const double numerator =
          a_i * tmp_i + a_j * tmp_j - (gamma - 1.) / 2. * (u_j - u_i);

      const double denominator =
          a_i * tmp_i * std::pow(p_i / p_j, -1. * (gamma - 1.0) / 2.0 / gamma) +
          a_j * tmp_j * 1.;

      return p_j * std::pow(numerator / denominator, 2. * gamma / (gamma - 1));
    }


    /**
     * For two given primitive states <code>riemann_data_i</code> and
     * <code>riemann_data_j</code>, and two guesses p_1 < p_2, compute
     * the gap in lambda between both guesses.
     *
     * See [1], page 914, (4.4a), (4.4b), (4.5), and (4.6)
     */
    inline DEAL_II_ALWAYS_INLINE std::array<double, 2>
    compute_gap(const double gamma,
                const std::array<double, 6> &riemann_data_i,
                const std::array<double, 6> &riemann_data_j,
                const double p_1,
                const double p_2)
    {
      const double nu_11 = lambda1_minus(gamma, riemann_data_i, p_2 /*SIC!*/);
      const double nu_12 = lambda1_minus(gamma, riemann_data_i, p_1 /*SIC!*/);

      const double nu_31 = lambda3_plus(gamma, riemann_data_j, p_1);
      const double nu_32 = lambda3_plus(gamma, riemann_data_j, p_2);

      const double lambda_max =
          std::max(positive_part(nu_32), negative_part(nu_11));

      const double gap =
          std::max(std::abs(nu_32 - nu_31), std::abs(nu_12 - nu_11));

      return {gap, lambda_max};
    }

  } /*anonymous namespace*/


  template <int dim>
  std::tuple<double, double, unsigned int>
  RiemannSolver<dim>::compute(const rank1_type &U_i,
                              const rank1_type &U_j,
                              const dealii::Tensor<1, dim> &n_ij) const
  {
    const double gamma = problem_description_->gamma();
    const double b = problem_description_->b();

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


  template <int dim>
  std::tuple<double, double, unsigned int>
  RiemannSolver<dim>::compute(const std::array<double, 6> &riemann_data_i,
                              const std::array<double, 6> &riemann_data_j) const
  {
    const double gamma = problem_description_->gamma();
    const double b = problem_description_->b();

    const double p_min = std::min(riemann_data_i[2], riemann_data_j[2]);
    const double p_max = std::max(riemann_data_i[2], riemann_data_j[2]);

    /*
     * Step 2: Shortcuts.
     *
     * In a number of cases we actually do not need to do a Newton search
     * for the optimal lambda upper bound, but know the answer right
     * away.
     */

    const double phi_p_min =
        phi(gamma, b, riemann_data_i, riemann_data_j, p_min);

    if (phi_p_min >= 0.) {
      const double p_star = 0.;
      const double lambda1 = lambda1_minus(gamma, riemann_data_i, p_star);
      const double lambda3 = lambda3_plus(gamma, riemann_data_j, p_star);
      const double lambda_max = std::max(std::abs(lambda1), std::abs(lambda3));
      return {lambda_max, p_star, 0};
    }

    const double phi_p_max =
        phi(gamma, b, riemann_data_i, riemann_data_j, p_max);

    // FIXME The == is a bit of a problem here
    if (phi_p_max == 0.) {
      const double p_star = p_max;
      const double lambda1 = lambda1_minus(gamma, riemann_data_i, p_star);
      const double lambda3 = lambda3_plus(gamma, riemann_data_j, p_star);
      const double lambda_max = std::max(std::abs(lambda1), std::abs(lambda3));
      return {lambda_max, p_star, 0};
    }

    /*
     * Step 3: Prepare quadratic Newton method.
     *
     * We need a good upper and lower bound, p_1 < p_star < p_2, for the
     * Newton method. (Ideally, for a moderate tolerance we might not
     * iterate at all.)
     */

    const double p_star_tilde =
        p_star_two_rarefaction(gamma, b, riemann_data_i, riemann_data_j);

    double p_1 = (phi_p_max < 0.) ? p_max : p_min;
    double p_2 =
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
      if (gap < eps_)
        return {lambda_max, p_2, i};

      /*
       * ... or if we reached the number of allowed Newton iteratoins.
       * lambda_max is a guaranteed upper bound, in the worst case we
       * overestimated the result.
       */
      if (i + 1 >= max_iter_)
        return {lambda_max, p_2, std::numeric_limits<unsigned int>::max()};

      /*
       * This is expensive:
       */

      const double phi_p_1 = phi(gamma, b, riemann_data_i, riemann_data_j, p_1);
      const double dphi_p_1 =
          dphi(gamma, b, riemann_data_i, riemann_data_j, p_1);
      const double phi_p_2 = phi(gamma, b, riemann_data_i, riemann_data_j, p_2);
      const double dphi_p_2 =
          dphi(gamma, b, riemann_data_i, riemann_data_j, p_2);

      /*
       * Sanity checks:
       *  * phi is monotone increasing and concave down: the derivative
       *    has to be positive, both function values have to be different
       *  * p_1 < p_2
       */

      Assert(dphi_p_1 > 0., dealii::ExcMessage("Houston, we are in trouble!"));
      Assert(dphi_p_2 > 0., dealii::ExcMessage("Houston, we are in trouble!"));
      Assert(phi_p_1 < phi_p_2,
             dealii::ExcMessage("Houston, we are in trouble!"));
      Assert(p_1 < p_2, dealii::ExcMessage("Houston, we are in trouble!"));

      /*
       * Compute divided differences
       */

      const double dd_11 = dphi_p_1;
      const double dd_12 = (phi_p_2 - phi_p_1) / (p_2 - p_1);
      const double dd_22 = dphi_p_2;

      const double dd_112 = (dd_12 - dd_11) / (p_2 - p_1);
      const double dd_122 = (dd_22 - dd_12) / (p_2 - p_1);

      /* Update left point: */
      const double discriminant_1 = dphi_p_1 * dphi_p_1 - 4. * phi_p_1 * dd_112;
      Assert(discriminant_1 > 0.,
             dealii::ExcMessage("Houston, we are in trouble!"));

      p_1 = p_1 - 2. * phi_p_1 / (dphi_p_1 + std::sqrt(discriminant_1));

      /* Update right point: */
      const double discriminant_2 = dphi_p_2 * dphi_p_2 - 4. * phi_p_2 * dd_122;
      Assert(discriminant_2 > 0.,
             dealii::ExcMessage("Houston, we are in trouble!"));

      p_2 = p_2 - 2. * phi_p_2 / (dphi_p_2 + std::sqrt(discriminant_2));

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

    } while (i++ < max_iter_);

    __builtin_unreachable();
  }


} /* namespace grendel */

#endif /* RIEMANN_SOLVER_TEMPLATE_H */
