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
  template <int dim, typename Number = double>
  class ProblemDescription final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * The dimension of the state space.
     */
    static constexpr unsigned int problem_dimension = 2 + dim;

    /**
     * An array holding all component names as a string.
     */
    const static std::array<std::string, dim + 2> component_names;

    /**
     * Underlyqing scalar number type. This typedef unpacks a
     * VectorizedArray and returns the underlying scalar number type
     * (either float, or double).
     */
    using ScalarNumber = typename get_value_type<Number>::type;

    /**
     * The storage type used for a state vector \f$\boldsymbol U\f$.
     */
    using rank1_type = dealii::Tensor<1, problem_dimension, Number>;

    /**
     * The storage type used for the flux \f$\mathbf{f}\f$.
     */
    using rank2_type =
        dealii::Tensor<1, problem_dimension, dealii::Tensor<1, dim, Number>>;

    /**
     * Constructor.
     */
    ProblemDescription(const std::string &subsection = "ProblemDescription");

    /**
     * @name ProblemDescription compile time options
     */
    //@{

    /**
     * Gamma \f$\gamma\f$.
     * @ingroup CompileTimeOptions
     */
    static constexpr ScalarNumber gamma = ScalarNumber(7./5.);

    /**
     * Covolume \f$b\f$.
     * @ingroup CompileTimeOptions
     */
    static constexpr ScalarNumber b = ScalarNumber(0.);
    static_assert(b == 0., "If you change this value, implement the rest...");

    //@}
    /**
     * @name Compile-time constant derived quantities
     */
    //@{

    /**
     * \f[
     *   \frac{1}{\gamma}
     * \f]
     */
    static constexpr ScalarNumber gamma_inverse = //
        ScalarNumber(1.) / gamma;

    /**
     * \f[
     *   \frac{1}{\gamma - 1}
     * \f]
     */
    static constexpr ScalarNumber gamma_minus_one_inverse =
        ScalarNumber(1.) / (gamma - ScalarNumber(1.));

    /**
     * \f[
     *   \frac{1}{\gamma + 1}
     * \f]
     */
    static constexpr ScalarNumber gamma_plus_one_inverse =
        ScalarNumber(1.) / (gamma + ScalarNumber(1.));

    /**
     * \f[
     *   \frac{\gamma - 1}{\gamma + 1}
     * \f]
     */
    static constexpr ScalarNumber gamma_minus_one_over_gamma_plus_one =
        (gamma - ScalarNumber(1.)) / (gamma + ScalarNumber(1.));

    //@}
    /**
     * @name Computing derived physical quantities.
     */
    //@{

    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, return
     * the momentum vector <code>[U[1], ..., U[1+dim]]</code>.
     */
    static dealii::Tensor<1, dim, Number> momentum(const rank1_type &U);


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the internal energy \f$\varepsilon = (\rho e)\f$.
     */
    static Number internal_energy(const rank1_type &U);


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the derivative of the internal energy
     * \f$\varepsilon = (\rho e)\f$.
     */
    static rank1_type internal_energy_derivative(const rank1_type &U);


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
    static Number pressure(const rank1_type &U);


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * the (physical) speed of sound:
     * \f[
     *   c^2 = \frac{\gamma * p}{\rho\;(1 - b * \rho)}
     * \f]
     */
    static Number speed_of_sound(const rank1_type &U);


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the (scaled) specific entropy
     * \f[
     *   e^{(\gamma-1)s} = \frac{\rho\,e}{\rho^\gamma}.
     * \f]
     */
    static Number specific_entropy(const rank1_type &U);


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the Harten-type entropy
     * \f[
     *   \eta = (\rho^2 e) ^ {1 / (\gamma + 1)}.
     * \f]
     */
    static Number harten_entropy(const rank1_type &U);


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the derivative \f$\eta'\f$ of the Harten-type entropy
     * \f[
     *   \eta = (\rho^2 e) ^ {1 / (\gamma + 1)}.
     * \f]
     */
    static rank1_type harten_entropy_derivative(const rank1_type &U);


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the entropy \f$\eta = p^{1/\gamma}\f$.
     */
    static Number mathematical_entropy(const rank1_type U);


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the derivative \f$\eta'\f$ of the entropy \f$\eta =
     * p^{1/\gamma}\f$.
     */
    static rank1_type mathematical_entropy_derivative(const rank1_type U);


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
    static rank2_type f(const rank1_type &U);

    //@}

  private:
    /**
     * @name Run time options
     */
    //@{

    std::string description_;
    ACCESSOR_READ_ONLY(description)

    Number mu_;
    ACCESSOR_READ_ONLY(mu)

    Number lambda_;
    ACCESSOR_READ_ONLY(lambda)

    Number kappa_;
    ACCESSOR_READ_ONLY(kappa)

    //@}
  };

  /* Inline definitions */

  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, dim, Number>
  ProblemDescription<dim, Number>::momentum(const rank1_type &U)
  {
    dealii::Tensor<1, dim, Number> result;
    for (unsigned int i = 0; i < dim; ++i)
      result[i] = U[1 + i];
    return result;
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  ProblemDescription<dim, Number>::internal_energy(const rank1_type &U)
  {
    /*
     * rho e = (E - 1/2*m^2/rho)
     */
    const Number rho_inverse = ScalarNumber(1.) / U[0];
    const auto m = momentum(U);
    const Number E = U[dim + 1];
    return E - ScalarNumber(0.5) * m.norm_square() * rho_inverse;
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline
      typename ProblemDescription<dim, Number>::rank1_type
      ProblemDescription<dim, Number>::internal_energy_derivative(
          const rank1_type &U)
  {
    /*
     * With
     *   rho e = E - 1/2 |m|^2 / rho
     * we get
     *   (rho e)' = (1/2m^2/rho^2, -m/rho , 1 )^T
     */

    const Number rho_inverse = ScalarNumber(1.) / U[0];
    const auto u = momentum(U) * rho_inverse;

    rank1_type result;

    result[0] = ScalarNumber(0.5) * u.norm_square();
    for (unsigned int i = 0; i < dim; ++i) {
      result[1 + i] = -u[i];
    }
    result[dim + 1] = ScalarNumber(1.);

    return result;
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  ProblemDescription<dim, Number>::pressure(const rank1_type &U)
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

    return (gamma - ScalarNumber(1.)) * internal_energy(U);
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  ProblemDescription<dim, Number>::speed_of_sound(const rank1_type &U)
  {
    /* c^2 = gamma * p / rho / (1 - b * rho) */
    const Number rho_inverse = ScalarNumber(1.) / U[0];
    const Number p = pressure(U);
    return std::sqrt(gamma * p * rho_inverse);
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  ProblemDescription<dim, Number>::specific_entropy(const rank1_type &U)
  {
    /*
     * We have
     *   exp((gamma - 1)s) = (rho e) / rho ^ gamma
     */
    const auto rho_inverse = ScalarNumber(1.) / U[0];
    return internal_energy(U) * ryujin::pow(rho_inverse, gamma);
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  ProblemDescription<dim, Number>::harten_entropy(const rank1_type &U)
  {
    /*
     * We have
     *   rho^2 e = \rho E - 1/2*m^2
     */
    const Number rho = U[0];
    const auto m = momentum(U);
    const Number E = U[dim + 1];

    const Number rho_rho_e = rho * E - ScalarNumber(0.5) * m.norm_square();
    return ryujin::pow(rho_rho_e, gamma_plus_one_inverse);
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline
      typename ProblemDescription<dim, Number>::rank1_type
      ProblemDescription<dim, Number>::harten_entropy_derivative(
          const rank1_type &U)
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
    const Number rho = U[0];
    const auto m = momentum(U);
    const Number E = U[dim + 1];

    const Number rho_rho_e = rho * E - ScalarNumber(0.5) * m.norm_square();

    const auto factor =
        gamma_plus_one_inverse *
        ryujin::pow(rho_rho_e, -gamma * gamma_plus_one_inverse);

    rank1_type result;

    result[0] = factor * E;
    for (unsigned int i = 0; i < dim; ++i) {
      result[1 + i] = -factor * m[i];
    }
    result[dim + 1] = factor * rho;

    return result;
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  ProblemDescription<dim, Number>::mathematical_entropy(const rank1_type U)
  {
    const auto p = pressure(U);
    return ryujin::pow(p, gamma_inverse);
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline
      typename ProblemDescription<dim, Number>::rank1_type
      ProblemDescription<dim, Number>::mathematical_entropy_derivative(
          const rank1_type U)
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

    const Number &rho = U[0];
    const Number rho_inverse = ScalarNumber(1.) / rho;
    const auto u = momentum(U) * rho_inverse;
    const auto p = pressure(U);

    const auto factor = (gamma - ScalarNumber(1.0)) * gamma_inverse *
                        ryujin::pow(p, gamma_inverse - ScalarNumber(1.));

    rank1_type result;

    result[0] = factor * ScalarNumber(0.5) * u.norm_square();
    result[dim + 1] = factor;
    for (unsigned int i = 0; i < dim; ++i) {
      result[1 + i] = -factor * u[i];
    }

    return result;
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline
      typename ProblemDescription<dim, Number>::rank2_type
      ProblemDescription<dim, Number>::f(const rank1_type &U)
  {
    const Number rho_inverse = ScalarNumber(1.) / U[0];
    const auto m = momentum(U);
    const auto p = pressure(U);
    const Number E = U[dim + 1];

    rank2_type result;

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
