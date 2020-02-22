#ifndef INITIAL_VALUES_H
#define INITIAL_VALUES_H

#include <compile_time_options.h>

#include "offline_data.h"
#include "problem_description.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <functional>

namespace grendel
{
  template <int dim, typename Number = double>
  class InitialValues : public dealii::ParameterAcceptor
  {
  public:
    static constexpr unsigned int problem_dimension =
        ProblemDescription<dim>::problem_dimension;

    static constexpr Number gamma = ProblemDescription<dim, Number>::gamma;

    static constexpr Number b = ProblemDescription<dim, Number>::b;

    using scalar_type = dealii::LinearAlgebra::distributed::Vector<Number>;
    using vector_type = std::array<scalar_type, problem_dimension>;

    using rank1_type = dealii::Tensor<1, problem_dimension, Number>;

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
    const std::function<rank1_type(const dealii::Point<dim> &point, Number t)>
        &initial_state;

    /**
     * Given a reference to an OfflineData object this routine computes and
     * returns a state vector populates with initial values for specified
     * time @p t.
     */
    vector_type interpolate(const OfflineData<dim, Number> &offline_data,
                            Number t = 0);

  private:
    std::string configuration_;

    dealii::Point<dim> initial_position_;
    dealii::Tensor<1, dim> initial_direction_;

    dealii::Tensor<1, 3, Number> initial_1d_state_;
    dealii::Tensor<1, 3, Number> initial_1d_state_contrast_;

    Number initial_mach_number_;

    Number initial_vortex_beta_;

    Number perturbation_;

    /*
     * Internal function object that we used to implement the
     * internal_state function for all internal states:
     */
    std::function<rank1_type(const dealii::Point<dim> &point, Number t)>
        initial_state_;
  };

} /* namespace grendel */

#endif /* INITIAL_VALUES_H */
