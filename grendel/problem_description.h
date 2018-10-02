#ifndef PROBLEM_DESCRIPTION_H
#define PROBLEM_DESCRIPTION_H

#include "boilerplate.h"

#include <deal.II/base/parameter_acceptor.h>
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
  class ProblemDescription : public dealii::ParameterAcceptor
  {
  public:

    /**
     * The dimension of the state space.
     */
    static constexpr unsigned int problem_dimension = 2 + dim;


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
     * Constructor.
     */
    ProblemDescription(const std::string &subsection = "ProblemDescription");


    /**
     * Destructor. We prevent the creation of derived classes by declaring
     * the destructor to be <code>final</code>.
     */
    virtual ~ProblemDescription() final = default;


    /**
     * Callback for ParameterAcceptor::initialize(). After we read in
     * configuration parameters from the parameter file we have to do some
     * (minor) preparatory work in this class to precompute some initial
     * state values. Do this with a callback.
     */
    void parse_parameters_callback();


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, return
     * the momentum vector <code>[U[1], ..., U[1+dim]]</code>.
     */
    static inline DEAL_II_ALWAYS_INLINE dealii::Tensor<1, dim>
    momentum_vector(const rank1_type &U);


    /**
     * Given a position @p point return the corresponding initial state.
     * This function is used to interpolate initial values.
     */
    rank1_type initial_state(const dealii::Point<dim> &point) const;


    /**
     * Given a state @p U compute <code>f(U)</code>.
     */
    inline DEAL_II_ALWAYS_INLINE rank2_type f(const rank1_type &U) const;


  protected:
    double gamma_;
    A_RO(gamma)

    double b_;
    A_RO(b)

    double cfl_constant_;
    A_RO(cfl_constant)

    rank1_type initial_shock_front_state_L_;
    A_RO(initial_shock_front_state_L)

    rank1_type initial_shock_front_state_R_;
    A_RO(initial_shock_front_state_R)

  private:
    std::string initial_state_;
    double initial_shock_front_mach_number_;
    dealii::Tensor<1, dim> initial_shock_front_direction_;
    dealii::Point<dim> initial_shock_front_position_;
  };


  /*
   * We have to provide the implementation of the following two inline
   * functions in this header file (otherwise they cannot be inlined). Both
   * functions are performance critical because they get evaluated
   * repeatedly during time stepping.
   */


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE dealii::Tensor<1, dim>
  ProblemDescription<dim>::momentum_vector(const rank1_type &U)
  {
    dealii::Tensor<1, dim> result;
    std::copy(&U[1], &U[1 + dim], &result[0]);
    return result;
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE typename ProblemDescription<dim>::rank2_type
  ProblemDescription<dim>::f(const rank1_type &U) const
  {
    const double rho = U[0];
    const auto m = momentum_vector(U);
    const double E = U[dim + 1];
    /*
     * With
     *   u = m / rho
     *   e = rho^-1 E - 1/2 |u|^2
     *   p(1-b rho) = (gamma - 1) e rho
     * We get: p = (gamma - 1)/(1 - b*rho)*(E - 1/2*m^2/rho)
     */
    const double p =
        (gamma_ - 1.) / (1. - b_ * rho) * (E - 0.5 * m.norm_square() / rho);

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
