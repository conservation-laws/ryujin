//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef PROBLEM_DESCRIPTION_H
#define PROBLEM_DESCRIPTION_H

#include <compile_time_options.h>

#include "convenience_macros.h"
#include "simd.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <array>
#include <functional>

namespace ryujin
{
  /**
   * Description of a @p dim dimensional hyperbolic conservation law.
   *
   * We have a (2 + dim) dimensional state space \f$[\rho, \textbf m,
   * E]\f$, where \f$\rho\f$ denotes the density, \f$\textbf m\f$ is the
   * momentum, and \f$E\f$ is the total energy.
   *
   * @ingroup EulerModule
   */
  class ProblemDescription final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * The dimension of the state space.
     */
    template <int dim>
    static constexpr unsigned int problem_dimension = 2 + dim;

    /**
     * An array holding all component names as a string.
     */
    template <int dim>
    static const std::array<std::string, dim + 2> component_names;

    /**
     * The storage type used for a state vector \f$\boldsymbol U\f$.
     */
    template <int dim, typename Number>
    using rank1_type = dealii::Tensor<1, problem_dimension<dim>, Number>;

    /**
     * The storage type used for the flux \f$\mathbf{f}\f$.
     */
    template <int dim, typename Number>
    using rank2_type = dealii::
        Tensor<1, problem_dimension<dim>, dealii::Tensor<1, dim, Number>>;

    /**
     * Constructor.
     */
    ProblemDescription(const std::string &subsection = "ProblemDescription");

    /**
     * Callback for ParameterAcceptor::initialize(). After we read in
     * configuration parameters from the parameter file we have to do some
     * (minor) preparatory work in this class to precompute some common
     * quantities. Do this with a callback.
     */
    void parse_parameters_callback();

    /**
     * @name Computing derived physical quantities.
     */
    //@{

    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, return
     * the momentum vector <code>[U[1], ..., U[1+dim]]</code>.
     */
    template <int problem_dim, typename Number>
    static dealii::Tensor<1, problem_dim - 2, Number>
    momentum(const dealii::Tensor<1, problem_dim, Number> &U);


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the internal energy \f$\varepsilon = (\rho e)\f$.
     */
    template <int problem_dim, typename Number>
    static Number
    internal_energy(const dealii::Tensor<1, problem_dim, Number> &U);


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the derivative of the internal energy
     * \f$\varepsilon = (\rho e)\f$.
     */
    template <int problem_dim, typename Number>
    static dealii::Tensor<1, problem_dim, Number>
    internal_energy_derivative(const dealii::Tensor<1, problem_dim, Number> &U);


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the pressure \f$p\f$.
     *
     * We assume that the pressure is given by a polytropic equation of
     * state, i.e.,
     * \f[
     *   p = \frac{\gamma - 1}{1 - b*\rho}\; (\rho e)
     * \f]
     */
    template <int problem_dim, typename Number>
    Number pressure(const dealii::Tensor<1, problem_dim, Number> &U) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * the (physical) speed of sound:
     * \f[
     *   c^2 = \frac{\gamma * p}{\rho\;(1 - b * \rho)}
     * \f]
     */
    template <int problem_dim, typename Number>
    Number
    speed_of_sound(const dealii::Tensor<1, problem_dim, Number> &U) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the (scaled) specific entropy
     * \f[
     *   e^{(\gamma-1)s} = \frac{\rho\,e}{\rho^\gamma}.
     * \f]
     */
    template <int problem_dim, typename Number>
    Number
    specific_entropy(const dealii::Tensor<1, problem_dim, Number> &U) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the Harten-type entropy
     * \f[
     *   \eta = (\rho^2 e) ^ {1 / (\gamma + 1)}.
     * \f]
     */
    template <int problem_dim, typename Number>
    Number
    harten_entropy(const dealii::Tensor<1, problem_dim, Number> &U) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the derivative \f$\eta'\f$ of the Harten-type entropy
     * \f[
     *   \eta = (\rho^2 e) ^ {1 / (\gamma + 1)}.
     * \f]
     */
    template <int problem_dim, typename Number>
    dealii::Tensor<1, problem_dim, Number> harten_entropy_derivative(
        const dealii::Tensor<1, problem_dim, Number> &U) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the entropy \f$\eta = p^{1/\gamma}\f$.
     */
    template <int problem_dim, typename Number>
    Number
    mathematical_entropy(const dealii::Tensor<1, problem_dim, Number> U) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the derivative \f$\eta'\f$ of the entropy \f$\eta =
     * p^{1/\gamma}\f$.
     */
    template <int problem_dim, typename Number>
    dealii::Tensor<1, problem_dim, Number> mathematical_entropy_derivative(
        const dealii::Tensor<1, problem_dim, Number> U) const;


    /**
     * Given a state @p U compute the flux
     * \f[
     * \begin{pmatrix}
     *   \textbf m \\
     *   \textbf v\otimes \textbf m + p\mathbb{I}_d \\
     *   \textbf v(E+p)
     * \end{pmatrix},
     * \f]
     */
    template <int problem_dim, typename Number>
    rank2_type<problem_dim - 2, Number>
    f(const dealii::Tensor<1, problem_dim, Number> &U) const;

    //@}

  private:
    /**
     * @name Run time options
     */
    //@{

    std::string description_;
    ACCESSOR_READ_ONLY(description)

    double gamma_;
    ACCESSOR_READ_ONLY(gamma)

    double b_;
    ACCESSOR_READ_ONLY(b)

    double mu_;
    ACCESSOR_READ_ONLY(mu)

    double lambda_;
    ACCESSOR_READ_ONLY(lambda)

    double cv_inverse_kappa_;
    ACCESSOR_READ_ONLY(cv_inverse_kappa)

    //@}
    /**
     * @name Precomputed scalar quantitites
     */
    //@{
    double gamma_inverse_;
    double gamma_plus_one_inverse_;

    //@}
  };

  /* Inline definitions */

  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim - 2, Number>
  ProblemDescription::momentum(const dealii::Tensor<1, problem_dim, Number> &U)
  {
    constexpr int dim = problem_dim - 2;

    dealii::Tensor<1, dim, Number> result;
    for (unsigned int i = 0; i < dim; ++i)
      result[i] = U[1 + i];
    return result;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number ProblemDescription::internal_energy(
      const dealii::Tensor<1, problem_dim, Number> &U)
  {
    /*
     * rho e = (E - 1/2*m^2/rho)
     */

    constexpr int dim = problem_dim - 2;
    using ScalarNumber = typename get_value_type<Number>::type;

    const Number rho_inverse = ScalarNumber(1.) / U[0];
    const auto m = momentum(U);
    const Number E = U[dim + 1];
    return E - ScalarNumber(0.5) * m.norm_square() * rho_inverse;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  ProblemDescription::internal_energy_derivative(
      const dealii::Tensor<1, problem_dim, Number> &U)
  {
    /*
     * With
     *   rho e = E - 1/2 |m|^2 / rho
     * we get
     *   (rho e)' = (1/2m^2/rho^2, -m/rho , 1 )^T
     */

    constexpr int dim = problem_dim - 2;
    using ScalarNumber = typename get_value_type<Number>::type;

    const Number rho_inverse = ScalarNumber(1.) / U[0];
    const auto u = momentum(U) * rho_inverse;

    dealii::Tensor<1, problem_dim, Number> result;

    result[0] = ScalarNumber(0.5) * u.norm_square();
    for (unsigned int i = 0; i < dim; ++i) {
      result[1 + i] = -u[i];
    }
    result[dim + 1] = ScalarNumber(1.);

    return result;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number ProblemDescription::pressure(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    /*
     * With
     *   u = m / rho
     *   e = rho^-1 E - 1/2 |u|^2
     *   p(1-b rho) = (gamma - 1) e rho
     * we get
     *   p = (gamma - 1)/(1 - b*rho) * (rho e)
     * (Here we have set b = 0)
     */

    using ScalarNumber = typename get_value_type<Number>::type;
    return (gamma_ - ScalarNumber(1.)) * internal_energy(U);
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number ProblemDescription::speed_of_sound(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    /* c^2 = gamma * p / rho / (1 - b * rho) */

    using ScalarNumber = typename get_value_type<Number>::type;

    const Number rho_inverse = ScalarNumber(1.) / U[0];
    const Number p = pressure(U);
    return std::sqrt(gamma_ * p * rho_inverse);
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number ProblemDescription::specific_entropy(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    /*
     * We have
     *   exp((gamma - 1)s) = (rho e) / rho ^ gamma
     */

    using ScalarNumber = typename get_value_type<Number>::type;

    const auto rho_inverse = ScalarNumber(1.) / U[0];
    return internal_energy(U) * ryujin::pow(rho_inverse, gamma_);
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number ProblemDescription::harten_entropy(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    /*
     * We have
     *   rho^2 e = \rho E - 1/2*m^2
     */

    constexpr int dim = problem_dim - 2;
    using ScalarNumber = typename get_value_type<Number>::type;

    const Number rho = U[0];
    const auto m = momentum(U);
    const Number E = U[dim + 1];

    const Number rho_rho_e = rho * E - ScalarNumber(0.5) * m.norm_square();
    return ryujin::pow(rho_rho_e, gamma_plus_one_inverse_);
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  ProblemDescription::harten_entropy_derivative(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    /*
     * With
     *   eta = (rho^2 e) ^ 1/(gamma+1)
     *   rho^2 e = rho * E - 1/2 |m|^2
     *
     * we get
     *
     *   eta' = 1/(gamma+1) * (rho^2 e) ^ -gamma/(gamma+1) * ( E , -m , rho )^T
     *
     * (Here we have set b = 0)
     */

    constexpr int dim = problem_dim - 2;
    using ScalarNumber = typename get_value_type<Number>::type;

    const Number rho = U[0];
    const auto m = momentum(U);
    const Number E = U[dim + 1];

    const Number rho_rho_e = rho * E - ScalarNumber(0.5) * m.norm_square();

    const auto factor =
        gamma_plus_one_inverse_ *
        ryujin::pow(rho_rho_e, -gamma_ * gamma_plus_one_inverse_);

    dealii::Tensor<1, problem_dim, Number> result;

    result[0] = factor * E;
    for (unsigned int i = 0; i < dim; ++i) {
      result[1 + i] = -factor * m[i];
    }
    result[dim + 1] = factor * rho;

    return result;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number ProblemDescription::mathematical_entropy(
      const dealii::Tensor<1, problem_dim, Number> U) const
  {
    const auto p = pressure(U);
    return ryujin::pow(p, gamma_inverse_);
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  ProblemDescription::mathematical_entropy_derivative(
      const dealii::Tensor<1, problem_dim, Number> U) const
  {
    /*
     * With
     *   eta = p ^ (1/gamma)
     *   p = (gamma - 1) * (rho e)
     *   rho e = E - 1/2 |m|^2 / rho
     *
     * we get
     *
     *   eta' = (gamma - 1)/gamma p ^(1/gamma - 1) *
     *
     *     (1/2m^2/rho^2 , -m/rho , 1 )^T
     *
     * (Here we have set b = 0)
     */

    constexpr int dim = problem_dim - 2;
    using ScalarNumber = typename get_value_type<Number>::type;

    const Number &rho = U[0];
    const Number rho_inverse = ScalarNumber(1.) / rho;
    const auto u = momentum(U) * rho_inverse;
    const auto p = pressure(U);

    const auto factor = (gamma_ - ScalarNumber(1.0)) * gamma_inverse_ *
                        ryujin::pow(p, gamma_inverse_ - ScalarNumber(1.));

    dealii::Tensor<1, problem_dim, Number> result;

    result[0] = factor * ScalarNumber(0.5) * u.norm_square();
    result[dim + 1] = factor;
    for (unsigned int i = 0; i < dim; ++i) {
      result[1 + i] = -factor * u[i];
    }

    return result;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline ProblemDescription::rank2_type<problem_dim - 2,
                                                              Number>
  ProblemDescription::f(const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    constexpr int dim = problem_dim - 2;
    using ScalarNumber = typename get_value_type<Number>::type;

    const Number rho_inverse = ScalarNumber(1.) / U[0];
    const auto m = momentum(U);
    const auto p = pressure(U);
    const Number E = U[dim + 1];

    rank2_type<dim, Number> result;

    result[0] = m;
    for (unsigned int i = 0; i < dim; ++i) {
      result[1 + i] = m * (m[i] * rho_inverse);
      result[1 + i][i] += p;
    }
    result[dim + 1] = m * (rho_inverse * (E + p));

    return result;
  }


} /* namespace ryujin */

#endif /* PROBLEM_DESCRIPTION_H */
