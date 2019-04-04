#ifndef PROBLEM_DESCRIPTION_H
#define PROBLEM_DESCRIPTION_H

#include "helper.h"

#include <deal.II/base/tensor.h>

#include <array>
#include <functional>

namespace grendel
{
  /**
   * The nD compressible Euler problem
   *
   * We have a (2 + n) dimensional state space [rho, m_1, ..., m_n, E],
   * where rho denotes the pressure, [m_1, ..., m_n] is the momentum vector
   * field, and E is the total energy.
   *
   * FIXME: Description
   */
  template <int dim>
  class ProblemDescription
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
     * Gamma.
     */
    static constexpr double gamma = 7. / 5.;


    /**
     * Covolume b.
     */
    static constexpr double b = 0.;
    static_assert(b == 0., "If you change this value, implement the rest...");


    /**
     * rank1_type denotes the storage type used for a state vector
     */
    typedef dealii::Tensor<1, problem_dimension> rank1_type;


    /**
     * rank2_type denotes the storage type used for the range of f.
     */
    typedef dealii::Tensor<1, problem_dimension, dealii::Tensor<1, dim>>
        rank2_type;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, return
     * the momentum vector <code>[U[1], ..., U[1+dim]]</code>.
     */
    static inline DEAL_II_ALWAYS_INLINE dealii::Tensor<1, dim>
    momentum(const rank1_type &U);


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the internal energy (\rho e).
     */
    static inline DEAL_II_ALWAYS_INLINE double
    internal_energy(const rank1_type &U);


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the derivative of the internal energy (\rho e).
     */
    static inline DEAL_II_ALWAYS_INLINE rank1_type
    internal_energy_derivative(const rank1_type &U);


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the pressure .
     */
    static inline DEAL_II_ALWAYS_INLINE double pressure(const rank1_type &U);


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the specific entropy
     * e^((\gamma-1)s) = (rho e) / rho ^ gamma.
     */
    static inline DEAL_II_ALWAYS_INLINE double
    specific_entropy(const rank1_type &U);


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the entropy \eta = p^(1/\gamma)
     */
    static inline DEAL_II_ALWAYS_INLINE double entropy(const rank1_type &U);


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the derivative \eta' of the entropy \eta = p^(1/\gamma)
     */
    static inline DEAL_II_ALWAYS_INLINE rank1_type
    entropy_derivative(const rank1_type &U);


    /**
     * Given a state @p U compute <code>f(U)</code>.
     */
    static inline DEAL_II_ALWAYS_INLINE rank2_type f(const rank1_type &U);
  };


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE dealii::Tensor<1, dim>
  ProblemDescription<dim>::momentum(const rank1_type &U)
  {
    dealii::Tensor<1, dim> result;
    std::copy(&U[1], &U[1 + dim], &result[0]);
    return result;
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE double
  ProblemDescription<dim>::internal_energy(const rank1_type &U)
  {
    /*
     * rho e = (E - 1/2*m^2/rho)
     */
    const double &rho = U[0];
    const auto m = momentum(U);
    const double &E = U[dim + 1];
    return E - 0.5 * m.norm_square() / rho;
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE typename ProblemDescription<dim>::rank1_type
  ProblemDescription<dim>::internal_energy_derivative(const rank1_type &U)
  {
    /*
     * With
     *   rho e = E - 1/2 |m|^2 / rho
     * we get
     *   (rho e)' = (1/2m^2/rho^2, -m/rho , 1 )^T
     */

    const double &rho = U[0];
    const auto u = momentum(U) / rho;

    rank1_type result;

    result[0] = 0.5 * u.norm_square();
    for (unsigned int i = 0; i < dim; ++i) {
      result[1 + i] = -u[i];
    }
    result[dim + 1] = 1.;

    return result;
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE double
  ProblemDescription<dim>::pressure(const rank1_type &U)
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

    return (gamma - 1.) * internal_energy(U);
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE double
  ProblemDescription<dim>::specific_entropy(const rank1_type &U)
  {
    /*
     * We have
     *   exp((gamma - 1)s) = (rho e) / rho ^ gamma
     */
    const auto &rho = U[0];
    return internal_energy(U) / std::pow(rho, gamma);
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE double
  ProblemDescription<dim>::entropy(const rank1_type &U)
  {
    const auto p = pressure(U);
    return std::pow(p, 1. / gamma);
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE typename ProblemDescription<dim>::rank1_type
  ProblemDescription<dim>::entropy_derivative(const rank1_type &U)
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

    const double &rho = U[0];
    const auto u = momentum(U) / rho;
    const auto p = pressure(U);

    const auto factor = (gamma - 1) / gamma * std::pow(p, 1. / gamma - 1.);

    rank1_type result;

    result[0] = factor * 1. / 2. * u.norm_square();
    result[dim + 1] = factor;
    for (unsigned int i = 0; i < dim; ++i) {
      result[1 + i] = -factor * u[i];
    }

    return result;
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE typename ProblemDescription<dim>::rank2_type
  ProblemDescription<dim>::f(const rank1_type &U)
  {
    const double &rho = U[0];
    const auto m = momentum(U);
    const auto p = pressure(U);
    const double &E = U[dim + 1];

    rank2_type result;

    result[0] = m;
    for (unsigned int i = 0; i < dim; ++i) {
      result[1 + i] = m * m[i] / rho;
      result[1 + i][i] += p;
    }
    result[dim + 1] = m / rho * (E + p);

    return result;
  }

} /* namespace grendel */

#endif /* PROBLEM_DESCRIPTION_H */
