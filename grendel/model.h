#ifndef Model_H
#define Model_H

#include "boilerplate.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <functional>

namespace grendel
{
  /**
   * Return the positive part of value number.
   */
  double positive_part(const double number)
  {
    return (std::abs(number) + number) / 2.0;
  }


  /**
   *
   * Return the negative part of value number.
   */
  double negative_part(const double number)
  {
    return (std::fabs(number) - number) / 2.0;
  }


  /**
   * The nD compressible Euler problem
   *
   * FIXME: Desciption
   *
   * We have a (2 + n) dimensional state space [rho, m_1, ..., m_n, E],
   * where rho denotes the pressure, [m_1, ..., m_n] is the momentum vector
   * field, and E is the total Energy.
   */
  template <int dim>
  class Model : public dealii::ParameterAcceptor
  {
  public:
    static constexpr unsigned int problem_dimension = 2 + dim;
    typedef dealii::Tensor<1, problem_dimension, double> rank1_type;

    Model(const std::string &subsection = "Model");
    virtual ~Model() final = default;

    /*
     * HERE BE DRAGONS!
     */

    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, return
     * the momentum vector <code>[U[1], ..., U[1+dim]]</code>.
     */
    static DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, dim>
    momentum_vector(const rank1_type &U)
    {
      dealii::Tensor<1, dim> result;
      std::copy(&U[1], &U[1 + dim], &result[0]);
      return result;
    }


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, and a
     * (normalized) "direction" n_ij, return the corresponding projected
     * state in the corresponding 1D Riemann problem.
     */
    static DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, 3>
    projected_state(const rank1_type &U, const dealii::Tensor<1, dim> &n_ij)
    {
      dealii::Tensor<1, 1 + 2> result;

      // rho:
      result[0] = U[0];

      // m:
      const auto m = momentum_vector(U);
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
    DEAL_II_ALWAYS_INLINE inline double
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
    DEAL_II_ALWAYS_INLINE inline double speed_of_sound_from_projected_state(
        const dealii::Tensor<1, 3> &projected_U) const
    {
      const auto rho = projected_U[0];
      const auto p = pressure_from_projected_state(projected_U);

      return std::sqrt(gamma_ * p / rho / (1.0 - b_ * rho));
    }


    /**
     * For a given projected state <code>projected_U</code> compute the
     * primitive state [rho_Z, u_Z, p_Z, a_Z, A_Z, B_Z]:
     *
     * FIXME: Describe state in more detail.
     */
    DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, 6>
    primitive_state_from_projected_state(
        const dealii::Tensor<1, 3> &projected_U) const
    {
      dealii::Tensor<1, 6> result;

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
    DEAL_II_ALWAYS_INLINE inline double
    f_Z(const dealii::Tensor<1, 6> &primitive_state, const double &p) const
    {
      const auto &rho_Z = primitive_state[0];
      const auto &p_Z = primitive_state[2];
      const auto &a_Z = primitive_state[3];
      const auto &A_Z = primitive_state[4];
      const auto &B_Z = primitive_state[5];

      if (p >= p_Z) {
        return (p - p_Z) * std::sqrt(A_Z / (p + B_Z));

      } else {

        const auto tmp = std::pow(p / p_Z, (gamma_ - 1.) / 2. / gamma_) - 1.;
        return 2. * a_Z * (1. - b_ * rho_Z) / (gamma_ - 1.) * tmp;
      }
    }


    /**
     * FIXME: Write a lengthy explanation.
     *
     * See [1], page 912, (3.4). FIXME find equation defining the
     */
    DEAL_II_ALWAYS_INLINE inline double
    df_Z(const dealii::Tensor<1, 6> &primitive_state, const double &p) const
    {
      const auto &rho_Z = primitive_state[0];
      const auto &p_Z = primitive_state[2];
      const auto &a_Z = primitive_state[3];
      const auto &A_Z = primitive_state[4];
      const auto &B_Z = primitive_state[5];

      if (p >= p_Z) {
        /* Derivative of (p - p_Z) * std::sqrt(A_Z / (p + B_Z)): */
        return std::sqrt(A_Z / (p + B_Z)) * (1. - 0.5 * (p - p_Z) / (p + B_Z));

      } else {

        /* Derivative of std::pow(p / p_Z, (gamma_ - 1.) / 2. / gamma_) - 1.*/
        const auto tmp = (gamma_ - 1.) / 2. / gamma_ *
                         std::pow(p / p_Z, (-1. - gamma_) / 2. / gamma_) / p_Z;
        return 2. * a_Z * (1. - b_ * rho_Z) / (gamma_ - 1.) * tmp;
      }
    }


    /**
     * FIXME: Write a lengthy explanation.
     *
     * See [1], page 912, (3.3).
     */
    DEAL_II_ALWAYS_INLINE inline double
    phi(const dealii::Tensor<1, 6> &primitive_state_i,
        const dealii::Tensor<1, 6> &primitive_state_j,
        const double &p) const
    {
      const auto &u_i = primitive_state_i[1];
      const auto &u_j = primitive_state_j[1];
      return f_Z(primitive_state_i, p) + f_Z(primitive_state_j, p) + u_i - u_j;
    }


    /**
     * FIXME: Write a lengthy explanation.
     *
     * See [1], page 912, (3.3). FIXME find equation defining the
     * derivative.
     */
    DEAL_II_ALWAYS_INLINE inline double
    dphi(const dealii::Tensor<1, 6> &primitive_state_i,
         const dealii::Tensor<1, 6> &primitive_state_j,
         const double &p) const
    {
      return df_Z(primitive_state_i, p) + df_Z(primitive_state_j, p);
    }


    /**
     * see [1], page 912, (3.7)
     */
    DEAL_II_ALWAYS_INLINE inline double
    lambda1_minus(const dealii::Tensor<1, 6> &primitive_state,
                  const double p_star) const
    {
      const auto &u_Z = primitive_state[1];
      const auto &p_Z = primitive_state[2];
      const auto &a_Z = primitive_state[3];

      const auto factor = (gamma_ + 1.0) / 2.0 / gamma_;
      const auto tmp = positive_part((p_star - p_Z) / p_Z);
      return u_Z - a_Z * std::sqrt(1.0 + factor * tmp);
    }


    /**
     * see [1], page 912, (3.8)
     */
    DEAL_II_ALWAYS_INLINE inline double
        lambda3_plus(const dealii::Tensor<1, 6> &primitive_state,
                     const double p_star) const
    {
      const auto &u_Z = primitive_state[1];
      const auto &p_Z = primitive_state[2];
      const auto &a_Z = primitive_state[3];

      const auto factor = (gamma_ + 1.0) / 2.0 / gamma_;
      const auto tmp = positive_part((p_star - p_Z) / p_Z);
      return u_Z + a_Z * std::sqrt(1.0 + factor * tmp);
    }


    /**
     * Two-rarefaction approximation to p_star computed for two primitive
     * states <code>primitive_state_i</code> and
     * <code>primitive_state_j</code>. See [1], page 914, (4.3)
     */
    DEAL_II_ALWAYS_INLINE inline double
    p_star_two_rarefaction(const dealii::Tensor<1, 6> &primitive_state_i,
                           const dealii::Tensor<1, 6> &primitive_state_j) const
    {
      const auto &rho_i = primitive_state_i[0];
      const auto &u_i = primitive_state_i[1];
      const auto &p_i = primitive_state_i[2];
      const auto &a_i = primitive_state_i[3];
      const auto &rho_j = primitive_state_j[0];
      const auto &u_j = primitive_state_j[1];
      const auto &p_j = primitive_state_j[2];
      const auto &a_j = primitive_state_j[3];

      /*
       * Notar bene (cf. [1, (4.3)]):
       *   a_Z^0 * sqrt(1 - b * rho_Z) = a_Z * (1 - b * rho_Z)
       * We have computed a_Z already, so we are simply going to use this
       * identity below:
       */

      const auto tmp_i = 1. - b_ * rho_i;
      const auto tmp_j = 1. - b_ * rho_j;

      // FIXME: Eliminate one std::pow call

      const auto numerator =
          a_i * tmp_i + a_j * tmp_j - (gamma_ - 1.) / 2. * (u_j - u_i);
      const auto denominator =
          a_i * tmp_i * std::pow(p_i, -1. * (gamma_ - 1.0) / 2.0 / gamma_) +
          a_j * tmp_j * std::pow(p_j, -1. * (gamma_ - 1.0) / 2.0 / gamma_);

      return std::pow(numerator / denominator, 2. * gamma_ / (gamma_ - 1));
    }


    /**
     * For two given primitive states <code>primitive_state_i</code> and
     * <code>primitive_state_j</code>, and two guesses p_1 < p_2, compute
     * the gap in lambda between both guesses.
     *
     * See [1], page 914, (4.4a), (4.4b), (4.5), and (4.6)
     */
    DEAL_II_ALWAYS_INLINE inline std::array<double, 2>
    compute_gap(const dealii::Tensor<1, 6> &primitive_state_i,
                const dealii::Tensor<1, 6> &primitive_state_j,
                const double p_1,
                const double p_2) const
    {
      const auto nu_11 = lambda1_minus(primitive_state_i, p_2 /*SIC!*/);
      const auto nu_12 = lambda1_minus(primitive_state_i, p_1 /*SIC!*/);

      const auto nu_31 = lambda3_plus(primitive_state_j, p_1);
      const auto nu_32 = lambda3_plus(primitive_state_j, p_2);

      const auto lambda_min =
          std::max(positive_part(nu_31), negative_part(nu_12));
      const auto lambda_max =
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


    /**
     * FIXME: Description
     *
     * For two given states U_i a U_j and a (normalized) "direction" n_ij
     * compute an estimation of an upper bound for lambda.
     *
     * See [1], page 915, Algorithm 1
     *
     * References:
     *   [1] J.-L. Guermond, B. Popov. Fast estimation from above fo the
     *       maximum wave speed in the Riemann problem for the Euler equations.
     */
    DEAL_II_ALWAYS_INLINE double
    lambda(const rank1_type &U_i,
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

      const auto p_min = std::min(primitive_state_i[0], primitive_state_j[0]);
      const auto p_max = std::max(primitive_state_i[0], primitive_state_j[0]);

      /*
       * Step 2: Shortcuts.
       *
       * In a number of cases we actually do not need to do a Newton search
       * for the optimal lambda upper bound, but know the answer right
       * away.
       */

      const auto phi_p_min = phi(primitive_state_i, primitive_state_j, p_min);

      if (phi_p_min >= 0.) {
        const double p_star = 0.;
        const auto lambda1 = lambda1_minus(primitive_state_i, p_star);
        const auto lambda3 = lambda3_plus(primitive_state_j, p_star);
        const auto lambda_max = std::max(std::abs(lambda1), std::abs(lambda3));
        return lambda_max;
      }

      const auto phi_p_max = phi(primitive_state_i, primitive_state_j, p_max);

      if(phi_p_max == 0.) {
        const auto p_star = p_max;
        const auto lambda1 = lambda1_minus(primitive_state_i, p_star);
        const auto lambda3 = lambda3_plus(primitive_state_j, p_star);
        const auto lambda_max = std::max(std::abs(lambda1), std::abs(lambda3));
        return lambda_max;
      }

      /*
       * Step 3: Prepare Newton secant method.
       *
       * We need a good upper and lower bound, p_1 < p_star < p_2, for the
       * Newton secant method. (Ideally, for a moderate tolerance we might
       * not iterate at all.)
       */

      const auto p_star_tilde =
          p_star_two_rarefaction(primitive_state_i, primitive_state_j);

      auto p_1 = (phi_p_max < 0.) ? p_max : p_min;
      auto p_2 =
          (phi_p_max < 0.) ? p_star_tilde : std::min(p_max, p_star_tilde);

      auto [gap, lambda_max] =
          compute_gap(primitive_state_i, primitive_state_j, p_1, p_2);

      /*
       * Step 4: Perform Newton secant iteration.
       */

      while(gap > eps_)
      {
        const auto phi_p_1 = phi(primitive_state_i, primitive_state_j, p_1);
        const auto dphi_p_1 = dphi(primitive_state_i, primitive_state_j, p_1);

        // phi is monote increasing and concave down, the derivative has to
        // be positive:
        Assert(dphi_p_1 <= 0.,
               dealii::ExcMessage("Houston, we are in trouble!"));

        /* Update left point with Newton step: */
        p_1 = p_1 - phi_p_1 / dphi_p_1;

        /* We have found our root (up to roundoff errros): */
        if (p_1 >= p_2)
          break;

        const auto phi_p_2 = phi(primitive_state_i, primitive_state_j, p_2);

        // phi is monote increasing and concave down, so both values have
        // to be different:
        Assert(phi_p_1 >= phi_p_2,
               dealii::ExcMessage("Houston, we are in trouble!"));

        /* Update right point with Secant method: */
        const auto slope = (phi_p_2 - phi_p_1) / (p_2 - p_1);
        p_2 = p_1 - phi_p_1 / slope;

        // FIXME
//         std::tie(gap, lambda_max) =
//             compute_gap(primitive_state_i, primitive_state_j, p_1, p_2);
      }

      return lambda_max;
    }

  private:
    double gamma_;
    double b_;

    double eps_;
  };

} /* namespace grendel */

#endif /* Model_H */
