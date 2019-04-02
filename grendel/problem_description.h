#ifndef PROBLEM_DESCRIPTION_H
#define PROBLEM_DESCRIPTION_H

#include "helper.h"

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
     * An array holding all component names as a string.
     */
    const static std::array<std::string, dim + 2> component_names;


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
    momentum(const rank1_type &U);


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the internal energy \rho e.
     */
    inline DEAL_II_ALWAYS_INLINE double
    internal_energy(const rank1_type &U) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the pressure .
     */
    inline DEAL_II_ALWAYS_INLINE double pressure(const rank1_type &U) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the entropy \eta = p^(1/\gamma)
     */
    inline DEAL_II_ALWAYS_INLINE double entropy(const rank1_type &U) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the derivative \eta' of the entropy \eta = p^(1/\gamma)
     */
    inline DEAL_II_ALWAYS_INLINE rank1_type
    entropy_derivative(const rank1_type &U) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the entropy flux u \eta = u p^(1/\gamma)
     */
    inline DEAL_II_ALWAYS_INLINE dealii::Tensor<1, dim>
    entropy_flux(const rank1_type &U) const;


    /**
     * Given a state @p U compute <code>f(U)</code>.
     */
    inline DEAL_II_ALWAYS_INLINE rank2_type f(const rank1_type &U) const;


    /**
     * Given a position @p point return the corresponding (conserved)
     * initial state. This function is used to interpolate initial values.
     *
     * The additional time parameter "t" is for validation purposes.
     * Sometimes we know the (analytic) solution of a test tube
     * configuration and want to compare the numerical computation against
     * it.
     */
    inline DEAL_II_ALWAYS_INLINE rank1_type
    initial_state(const dealii::Point<dim> &point, double t) const;


  protected:
    double gamma_;
    ACCESSOR_READ_ONLY(gamma)

    double cfl_update_;
    ACCESSOR_READ_ONLY(cfl_update)

    double cfl_max_;
    ACCESSOR_READ_ONLY(cfl_max)

  private:
    std::string initial_state_;

    dealii::Tensor<1, dim> initial_direction_;
    dealii::Point<dim> initial_position_;
    double initial_mach_number_;

    double initial_vortex_beta_;

    /*
     * Internal function object that we used to implement the
     * internal_state function for all internal states:
     */
    std::function<rank1_type(const dealii::Point<dim> &point, double t)>
        initial_state_internal;
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
  ProblemDescription<dim>::internal_energy(const rank1_type &U) const
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
  inline DEAL_II_ALWAYS_INLINE double
  ProblemDescription<dim>::pressure(const rank1_type &U) const
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

    return (gamma_ - 1.) * internal_energy(U);
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE double
  ProblemDescription<dim>::entropy(const rank1_type &U) const
  {
    const auto p = pressure(U);
    return std::pow(p, 1. / gamma_);
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE typename ProblemDescription<dim>::rank1_type
  ProblemDescription<dim>::entropy_derivative(const rank1_type &U) const
  {
    /*
     * With
     *   eta = p ^ (1/gamma)
     *   p = (gamma - 1) * (rho e)
     *   rho e = E - 1/2 |m|^2 / rho
     *
     * we get
     *
     *   eta' = 1/gamma p ^(1/gamma - 1) *
     *
     *     (1/2m^2/rho^2 , rho m , 1 )^T
     *
     * (Here we have set b = 0)
     */

    const double &rho = U[0];
    const auto m = momentum(U);
    const auto p = pressure(U);

    const auto factor = 1 / gamma_ * std::pow(p, 1. / gamma_ - 1.);

    rank1_type result;

    result[0] = factor * 1. / 2. * m.norm_square() / rho / rho;
    result[dim + 1] = factor;
    for (unsigned int i = 0; i < dim; ++i) {
      result[1 + i] = factor * rho * m[i];
    }

    return result;
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE dealii::Tensor<1, dim>
  ProblemDescription<dim>::entropy_flux(const rank1_type &U) const
  {
    const auto &rho = U[0];
    const auto eta = entropy(U);
    const auto m = momentum(U);

    return eta * m / rho;
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE typename ProblemDescription<dim>::rank2_type
  ProblemDescription<dim>::f(const rank1_type &U) const
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


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE typename ProblemDescription<dim>::rank1_type
  ProblemDescription<dim>::initial_state(const dealii::Point<dim> &point,
                                         double t) const
  {
    return initial_state_internal(point, t);
  }

} /* namespace grendel */

#endif /* PROBLEM_DESCRIPTION_H */
