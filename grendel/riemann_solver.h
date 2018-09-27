#ifndef RIEMANN_SOLVER_H
#define RIEMANN_SOLVER_H

#include "boilerplate.h"
#include "problem_description.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <functional>

namespace grendel
{
  namespace
  {
    /**
     * Return the positive part of value number.
     */
    inline DEAL_II_ALWAYS_INLINE double positive_part(const double number)
    {
      return (std::abs(number) + number) / 2.0;
    }


    /**
     *
     * Return the negative part of value number.
     */
    inline DEAL_II_ALWAYS_INLINE double negative_part(const double number)
    {
      return (std::fabs(number) - number) / 2.0;
    }
  }


  /**
   * A fast approximative Riemann problem solver for the nD compressible
   * Euler problem.
   *
   * FIXME: Desciption
   */
  template <int dim>
  class RiemannSolver : public dealii::ParameterAcceptor
  {
  public:
    static constexpr unsigned int problem_dimension =
        ProblemDescription<dim>::problem_dimension;

    using rank1_type = typename ProblemDescription<dim>::rank1_type;

    RiemannSolver(const grendel::ProblemDescription<dim> &problem_description,
                  const std::string &subsection = "RiemannSolver");

    virtual ~RiemannSolver() final = default;


    /*
     * HERE BE DRAGONS!
     */


  private:
    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, and a
     * (normalized) "direction" n_ij, return the corresponding projected
     * state in the corresponding 1D Riemann problem.
     */
    static inline DEAL_II_ALWAYS_INLINE dealii::Tensor<1, 3>
    projected_state(const rank1_type &U, const dealii::Tensor<1, dim> &n_ij)
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
    pressure_from_projected_state(const dealii::Tensor<1, 3> &projected_U) const
    {
      return (gamma_ - 1.0) *
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
    inline DEAL_II_ALWAYS_INLINE double speed_of_sound_from_projected_state(
        const dealii::Tensor<1, 3> &projected_U) const
    {
      const double rho = projected_U[0];
      const double p = pressure_from_projected_state(projected_U);

      return std::sqrt(gamma_ * p / rho / (1.0 - b_ * rho));
    }


    /**
     * For a given projected state <code>projected_U</code> compute the
     * primitive state [rho_Z, u_Z, p_Z, a_Z, A_Z, B_Z]:
     *
     * FIXME: Describe state in more detail.
     */
    inline DEAL_II_ALWAYS_INLINE std::array<double, 6>
    primitive_state_from_projected_state(
        const dealii::Tensor<1, 3> &projected_U) const
    {
      std::array<double, 6> result;

      // rho_Z:
      result[0] = projected_U[0];
      // u_Z:
      result[1] = projected_U[1] / projected_U[0];
      // p_Z:
      result[2] = pressure_from_projected_state(projected_U);
      // a_Z:
      result[3] = speed_of_sound_from_projected_state(projected_U);
      // A_Z:
      result[4] =
          2.0 * (1.0 - b_ * projected_U[0]) / (gamma_ + 1.0) / projected_U[0];
      // B_Z:
      result[5] = (gamma_ - 1.0) / (gamma_ + 1.0) * result[2];

      return result;
    }


    /**
     * FIXME: Write a lengthy explanation.
     *
     * See [1], page 912, (3.4).
     */
    inline DEAL_II_ALWAYS_INLINE double
    f_Z(const std::array<double, 6> &primitive_state, const double &p) const
    {
      const auto &[rho_Z, u_Z, p_Z, a_Z, A_Z, B_Z] = primitive_state;

      if (p >= p_Z) {
        return (p - p_Z) * std::sqrt(A_Z / (p + B_Z));

      } else {

        const double tmp = std::pow(p / p_Z, (gamma_ - 1.) / 2. / gamma_) - 1.;
        return 2. * a_Z * (1. - b_ * rho_Z) / (gamma_ - 1.) * tmp;
      }
    }


    /**
     * FIXME: Write a lengthy explanation.
     *
     * See [1], page 912, (3.4). FIXME find equation defining the
     */
    inline DEAL_II_ALWAYS_INLINE double
    df_Z(const std::array<double, 6> &primitive_state, const double &p) const
    {
      const auto &[rho_Z, u_Z, p_Z, a_Z, A_Z, B_Z] = primitive_state;

      if (p >= p_Z) {
        /* Derivative of (p - p_Z) * std::sqrt(A_Z / (p + B_Z)): */
        return std::sqrt(A_Z / (p + B_Z)) * (1. - 0.5 * (p - p_Z) / (p + B_Z));

      } else {

        /* Derivative of std::pow(p / p_Z, (gamma_ - 1.) / 2. / gamma_) - 1.*/
        const double tmp = (gamma_ - 1.) / 2. / gamma_ *
                           std::pow(p / p_Z, (-1. - gamma_) / 2. / gamma_) /
                           p_Z;
        return 2. * a_Z * (1. - b_ * rho_Z) / (gamma_ - 1.) * tmp;
      }
    }


    /**
     * FIXME: Write a lengthy explanation.
     *
     * See [1], page 912, (3.3).
     */
    inline DEAL_II_ALWAYS_INLINE double
    phi(const std::array<double, 6> &primitive_state_i,
        const std::array<double, 6> &primitive_state_j,
        const double &p) const
    {
      const double &u_i = primitive_state_i[1];
      const double &u_j = primitive_state_j[1];

      return f_Z(primitive_state_i, p) + f_Z(primitive_state_j, p) + u_j - u_i;
    }


    /**
     * FIXME: Write a lengthy explanation.
     *
     * See [1], page 912, (3.3). FIXME find equation defining the
     * derivative.
     */
    inline DEAL_II_ALWAYS_INLINE double
    dphi(const std::array<double, 6> &primitive_state_i,
         const std::array<double, 6> &primitive_state_j,
         const double &p) const
    {
      return df_Z(primitive_state_i, p) + df_Z(primitive_state_j, p);
    }


    /**
     * see [1], page 912, (3.7)
     */
    inline DEAL_II_ALWAYS_INLINE double
    lambda1_minus(const std::array<double, 6> &primitive_state,
                  const double p_star) const
    {
      const auto &[rho_Z, u_Z, p_Z, a_Z, A_Z, B_Z] = primitive_state;

      const double factor = (gamma_ + 1.0) / 2.0 / gamma_;
      const double tmp = positive_part((p_star - p_Z) / p_Z);
      return u_Z - a_Z * std::sqrt(1.0 + factor * tmp);
    }


    /**
     * see [1], page 912, (3.8)
     */
    inline DEAL_II_ALWAYS_INLINE double
    lambda3_plus(const std::array<double, 6> &primitive_state,
                 const double p_star) const
    {
      const auto &[rho_Z, u_Z, p_Z, a_Z, A_Z, B_Z] = primitive_state;

      const double factor = (gamma_ + 1.0) / 2.0 / gamma_;
      const double tmp = positive_part((p_star - p_Z) / p_Z);
      return u_Z + a_Z * std::sqrt(1.0 + factor * tmp);
    }


    /**
     * Two-rarefaction approximation to p_star computed for two primitive
     * states <code>primitive_state_i</code> and
     * <code>primitive_state_j</code>. See [1], page 914, (4.3)
     */
    inline DEAL_II_ALWAYS_INLINE double
    p_star_two_rarefaction(const std::array<double, 6> &primitive_state_i,
                           const std::array<double, 6> &primitive_state_j) const
    {
      const auto &[rho_i, u_i, p_i, a_i, A_i, B_i] = primitive_state_i;
      const auto &[rho_j, u_j, p_j, a_j, A_j, B_j] = primitive_state_j;

      /*
       * Notar bene (cf. [1, (4.3)]):
       *   a_Z^0 * sqrt(1 - b * rho_Z) = a_Z * (1 - b * rho_Z)
       * We have computed a_Z already, so we are simply going to use this
       * identity below:
       */

      const double tmp_i = 1. - b_ * rho_i;
      const double tmp_j = 1. - b_ * rho_j;

      const double numerator =
          a_i * tmp_i + a_j * tmp_j - (gamma_ - 1.) / 2. * (u_j - u_i);

      const double denominator =
          a_i * tmp_i *
              std::pow(p_i / p_j, -1. * (gamma_ - 1.0) / 2.0 / gamma_) +
          a_j * tmp_j * 1.;

      return p_j *
             std::pow(numerator / denominator, 2. * gamma_ / (gamma_ - 1));
    }


    /**
     * For two given primitive states <code>primitive_state_i</code> and
     * <code>primitive_state_j</code>, and two guesses p_1 < p_2, compute
     * the gap in lambda between both guesses.
     *
     * See [1], page 914, (4.4a), (4.4b), (4.5), and (4.6)
     */
    inline DEAL_II_ALWAYS_INLINE std::array<double, 2>
    compute_gap(const std::array<double, 6> &primitive_state_i,
                const std::array<double, 6> &primitive_state_j,
                const double p_1,
                const double p_2) const
    {
      const double nu_11 = lambda1_minus(primitive_state_i, p_2 /*SIC!*/);
      const double nu_12 = lambda1_minus(primitive_state_i, p_1 /*SIC!*/);

      const double nu_31 = lambda3_plus(primitive_state_j, p_1);
      const double nu_32 = lambda3_plus(primitive_state_j, p_2);

      const double lambda_min =
          std::max(positive_part(nu_31), negative_part(nu_12));
      const double lambda_max =
          std::max(positive_part(nu_32), negative_part(nu_11));

      /*
       * We have to deal with the fact that lambda_min >= lambda_max due to
       * round-off errors. In this case, accept the guess and report a gap
       * of size 0.
       */
      if (lambda_min >= lambda_max)
        return {0., lambda_max};

      /*
       * In case lambda_min <= 0. we haven't converged. Just return a large
       * value to continue iterating.
       */
      if (lambda_min <= 0.0)
        return {1., lambda_max};

      return {lambda_max / lambda_min - 1.0, lambda_max};
    }

  public:
    /**
     * FIXME: Description
     *
     * For two given states U_i a U_j and a (normalized) "direction" n_ij
     * compute an estimation of an upper bound for lambda.
     *
     * See [1], page 915, Algorithm 1
     *
     * Returns a tuple consisting of lambda max and the number of Newton
     * iterations used in the solver to find it.
     *
     * References:
     *   [1] J.-L. Guermond, B. Popov. Fast estimation from above fo the
     *       maximum wave speed in the Riemann problem for the Euler equations.
     */
    std::tuple<double /*lambda_max*/, unsigned int /*iteration*/>
    lambda_max(const rank1_type &U_i,
               const rank1_type &U_j,
               const dealii::Tensor<1, dim> &n_ij) const
    {
      /*
       * Step 1: Compute projected 1D states and phi.
       */

      const auto projected_U_i = projected_state(U_i, n_ij);
      const auto projected_U_j = projected_state(U_j, n_ij);

      const auto primitive_state_i =
          primitive_state_from_projected_state(projected_U_i);
      const auto primitive_state_j =
          primitive_state_from_projected_state(projected_U_j);

      const double p_min = std::min(primitive_state_i[2], primitive_state_j[2]);
      const double p_max = std::max(primitive_state_i[2], primitive_state_j[2]);

      /*
       * Step 2: Shortcuts.
       *
       * In a number of cases we actually do not need to do a Newton search
       * for the optimal lambda upper bound, but know the answer right
       * away.
       */

      const double phi_p_min = phi(primitive_state_i, primitive_state_j, p_min);

      if (phi_p_min >= 0.) {
        const double p_star = 0.;
        const double lambda1 = lambda1_minus(primitive_state_i, p_star);
        const double lambda3 = lambda3_plus(primitive_state_j, p_star);
        const double lambda_max =
            std::max(std::abs(lambda1), std::abs(lambda3));
        return {lambda_max, 0};
      }

      const double phi_p_max = phi(primitive_state_i, primitive_state_j, p_max);

      if (phi_p_max == 0.) {
        const double p_star = p_max;
        const double lambda1 = lambda1_minus(primitive_state_i, p_star);
        const double lambda3 = lambda3_plus(primitive_state_j, p_star);
        const double lambda_max =
            std::max(std::abs(lambda1), std::abs(lambda3));
        return {lambda_max, 0};
      }

      /*
       * Step 3: Prepare quadratic Newton method.
       *
       * We need a good upper and lower bound, p_1 < p_star < p_2, for the
       * Newton method. (Ideally, for a moderate tolerance we might not
       * iterate at all.)
       */

      const double p_star_tilde =
          p_star_two_rarefaction(primitive_state_i, primitive_state_j);

      double p_1 = (phi_p_max < 0.) ? p_max : p_min;
      double p_2 =
          (phi_p_max < 0.) ? p_star_tilde : std::min(p_max, p_star_tilde);

      /*
       * Step 4: Perform quadratic Newton iteration.
       *
       * See [1], p. 915f (4.8) and (4.9)
       */

      for (unsigned int i = 0; i < max_iter_; ++i) {
        const auto [gap, lambda_max] =
            compute_gap(primitive_state_i, primitive_state_j, p_1, p_2);

        if (gap < eps_)
          return {lambda_max, i};

        /*
         * This is expensive:
         */

        const double phi_p_1 = phi(primitive_state_i, primitive_state_j, p_1);
        const double dphi_p_1 = dphi(primitive_state_i, primitive_state_j, p_1);
        const double phi_p_2 = phi(primitive_state_i, primitive_state_j, p_2);
        const double dphi_p_2 = dphi(primitive_state_i, primitive_state_j, p_2);

        /*
         * Sanity checks:
         *  * phi is monotone increasing and concave down: the derivative
         *    has to be positive, both function values have to be different
         *  * p_1 < p_2
         */

        Assert(dphi_p_1 > 0.,
               dealii::ExcMessage("Houston, we are in trouble!"));
        Assert(dphi_p_2 > 0.,
               dealii::ExcMessage("Houston, we are in trouble!"));
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
        const double discriminant_1 =
            dphi_p_1 * dphi_p_1 - 4. * phi_p_1 * dd_112;
        Assert(discriminant_1 > 0.,
               dealii::ExcMessage("Houston, we are in trouble!"));

        p_1 = p_1 - 2. * phi_p_1 / (dphi_p_1 + std::sqrt(discriminant_1));

        /* Update right point: */
        const double discriminant_2 =
            dphi_p_2 * dphi_p_2 - 4. * phi_p_2 * dd_122;
        Assert(discriminant_2 > 0.,
               dealii::ExcMessage("Houston, we are in trouble!"));

        p_2 = p_2 - 2. * phi_p_2 / (dphi_p_2 + std::sqrt(discriminant_2));


        /* We have found our root (up to roundoff erros): */
        if (p_1 >= p_2)
          return {lambda_max, i + 1};
      }

      AssertThrow(false,
                  dealii::ExcMessage("Newton method did not converge."));
      return {0., std::numeric_limits<unsigned int>::max()};
    }

  protected:
    dealii::SmartPointer<const grendel::ProblemDescription<dim>>
        problem_description_;
    A_RO(problem_description)

  private:
    const double &gamma_;
    const double &b_;

    double eps_;
    unsigned int max_iter_;
  };

} /* namespace grendel */

#endif /* RIEMANN_SOLVER_H */
