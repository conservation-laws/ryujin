#ifndef TIME_STEP_H
#define TIME_STEP_H

#include "helper.h"
#include "simd.h"

#include "limiter.h"
#include "matrix_communicator.h"

#include "initial_values.h"
#include "offline_data.h"
#include "problem_description.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/sparse_matrix.templates.h>
#include <deal.II/lac/vector.h>

namespace grendel
{

  template <int dim, typename Number = double>
  class TimeStep : public dealii::ParameterAcceptor
  {
  public:
    static constexpr unsigned int problem_dimension =
        ProblemDescription<dim, Number>::problem_dimension;

    using rank1_type = typename ProblemDescription<dim, Number>::rank1_type;
    using rank2_type = typename ProblemDescription<dim, Number>::rank2_type;

    typedef std::array<dealii::LinearAlgebra::distributed::Vector<Number>,
                       problem_dimension>
        vector_type;

    TimeStep(const MPI_Comm &mpi_communicator,
             dealii::TimerOutput &computing_timer,
             const grendel::OfflineData<dim, Number> &offline_data,
             const grendel::InitialValues<dim, Number> &initial_values,
             const std::string &subsection = "TimeStep");

    virtual ~TimeStep() final = default;

    void prepare();

    /**
     * Given a reference to a previous state vector U perform an explicit
     * euler step (and store the result in U). The function
     *
     *  - returns the computed maximal time step size tau_max
     *
     *  - performs a time step and populates the vector U_new by the
     *    result. The time step is performed with either tau_max (if tau ==
     *    0), or tau (if tau != 0). Here, tau_max is the computed maximal
     *    time step size and tau is the optional third parameter.
     */
    Number euler_step(vector_type &U, Number t, Number tau = 0.);

    /**
     * Given a reference to a previous state vector U perform an explicit
     * Heun 2nd order step (and store the result in U).
     *
     *  - returns the computed maximal time step size tau_max
     *
     * [Shu & Osher, Efficient Implementation of Essentially
     * Non-oscillatory Shock-Capturing Schemes JCP 77:439-471 (1988), Eq.
     * 2.15]
     */
    Number ssph2_step(vector_type &U, Number t);

    /**
     * Given a reference to a previous state vector U perform an explicit
     * SSP Runge Kutta 3rd order step (and store the result in U).
     *
     *  - returns the computed maximal time step size tau_max
     *
     * [Shu & Osher, Efficient Implementation of Essentially
     * Non-oscillatory Shock-Capturing Schemes JCP 77:439-471 (1988), Eq.
     * 2.18]
     */
    Number ssprk3_step(vector_type &U, Number t);

    /**
     * Given a reference to a previous state vector U perform an explicit
     * time step (and store the result in U). The function returns the
     * chosen time step size tau.
     *
     * Depending on the approximation order (first or second order) this
     * function chooses the 2nd order Heun, or the third order Runge Kutta
     * time stepping scheme.
     */
    Number step(vector_type &U, Number t);

    /* Options: */

    static constexpr enum class Order {
      first_order,
      second_order
    } order_ = Order::second_order;

    static constexpr enum class TimeStepOrder {
      first_order,
      second_order,
      third_order
    } time_step_order_ = TimeStepOrder::second_order;

    static constexpr unsigned int limiter_iter_ = 2;

  private:
    const MPI_Comm &mpi_communicator_;
    dealii::TimerOutput &computing_timer_;

    dealii::SmartPointer<const grendel::OfflineData<dim, Number>> offline_data_;
    dealii::SmartPointer<const grendel::InitialValues<dim, Number>>
        initial_values_;

    /* Scratch data: */

    dealii::SparseMatrix<Number> dij_matrix_;

    dealii::LinearAlgebra::distributed::Vector<Number> rho_second_variation_;
    dealii::LinearAlgebra::distributed::Vector<Number> rho_relaxation_;
    dealii::LinearAlgebra::distributed::Vector<Number> alpha_;
    ACCESSOR_READ_ONLY(alpha)

    typename Limiter<dim, Number>::vector_type bounds_;

    std::vector<unsigned int> transposed_indices;

    vector_type r_;

    dealii::SparseMatrix<Number> lij_matrix_;
    MatrixCommunicator<dim, Number> lij_matrix_communicator_;

    std::array<dealii::SparseMatrix<Number>, problem_dimension> pij_matrix_;

    vector_type temp_euler_;
    vector_type temp_ssprk_;

    /* Options: */

    Number cfl_update_;
    Number cfl_max_;
  };

} /* namespace grendel */

#endif /* TIME_STEP_H */
