#ifndef INITIAL_VALUES_H
#define INITIAL_VALUES_H

#include "problem_description.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <functional>

namespace grendel
{
  template <int dim>
  class InitialValues : public dealii::ParameterAcceptor
  {
  public:
    static constexpr unsigned int problem_dimension =
        ProblemDescription<dim>::problem_dimension;

    static constexpr double gamma = ProblemDescription<dim>::gamma;

    static constexpr double b = ProblemDescription<dim>::b;


    typedef dealii::Tensor<1, problem_dimension> rank1_type;


    /**
     * Constructor.
     */
    InitialValues(const std::string &subsection = "InitialValues");


    /**
     * Callback for ParameterAcceptor::initialize(). After we read in
     * configuration parameters from the parameter file we have to do some
     * (minor) preparatory work in this class to precompute some initial
     * state values. Do this with a callback.
     */
    void parse_parameters_callback();


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


  private:
    std::string configuration_;

    dealii::Point<dim> initial_position_;
    dealii::Tensor<1, dim> initial_direction_;

    dealii::Tensor<1, 3> initial_1d_state_;
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
  inline DEAL_II_ALWAYS_INLINE typename InitialValues<dim>::rank1_type
  InitialValues<dim>::initial_state(const dealii::Point<dim> &point,
                                         double t) const
  {
    return initial_state_internal(point, t);
  }

} /* namespace grendel */

#endif /* INITIAL_VALUES_H */
