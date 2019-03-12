#ifndef TIME_STEP_H
#define TIME_STEP_H

#include "helper.h"
#include "high_order.h"
#include "offline_data.h"
#include "problem_description.h"
#include "riemann_solver.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_vector.templates.h>
#include <deal.II/lac/vector.templates.h>

namespace grendel
{

  template <int dim>
  class TimeStep : public dealii::ParameterAcceptor
  {
  public:
    static constexpr unsigned int problem_dimension =
        ProblemDescription<dim>::problem_dimension;

    using rank1_type = typename ProblemDescription<dim>::rank1_type;
    using rank2_type = typename ProblemDescription<dim>::rank2_type;

    typedef std::array<dealii::LinearAlgebra::distributed::Vector<double>,
                       problem_dimension>
        vector_type;

    TimeStep(const MPI_Comm &mpi_communicator,
             dealii::TimerOutput &computing_timer,
             const grendel::OfflineData<dim> &offline_data,
             const grendel::ProblemDescription<dim> &problem_description,
             const grendel::RiemannSolver<dim> &riemann_solver,
             const grendel::HighOrder<dim> &high_order,
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
    double euler_step(vector_type &U, double tau = 0.);

    /**
     * Given a reference to a previous state vector U compute
     * perform an explicit SSP RK(3) step (and store the result in U).
     *
     * [Shu & Osher, Efficient Implementation of Essentially
     * Non-oscillatory Shock-Capturing Schemes JCP 77:439-471 (1988), Eq.
     * 2.18]
     */
    double ssprk_step(vector_type &U);

    /**
     * Given a reference to a previous state vector U perform an explicit
     * time step (and store the result in U). The function returns the
     * chosen time step.
     */
    double step(vector_type &U);

    /* Options: */

    static constexpr enum class Order {
      first_order,
      second_order
    } order_ = Order::second_order;

  protected:
    const MPI_Comm &mpi_communicator_;
    dealii::TimerOutput &computing_timer_;

    dealii::SmartPointer<const grendel::OfflineData<dim>> offline_data_;
    ACCESSOR_READ_ONLY(offline_data)

    dealii::SmartPointer<const grendel::ProblemDescription<dim>>
        problem_description_;
    ACCESSOR_READ_ONLY(problem_description)

    dealii::SmartPointer<const grendel::RiemannSolver<dim>> riemann_solver_;
    ACCESSOR_READ_ONLY(riemann_solver)

    dealii::SmartPointer<const grendel::HighOrder<dim>> high_order_;
    ACCESSOR_READ_ONLY(high_order)

  private:
    /* Scratch data: */

    dealii::SparseMatrix<double> dij_matrix_;

    dealii::LinearAlgebra::distributed::Vector<double> alpha_;
    ACCESSOR_READ_ONLY(alpha)

    dealii::LinearAlgebra::distributed::Vector<double> alpha_second_;
    ACCESSOR_READ_ONLY(alpha_second)

    dealii::LinearAlgebra::distributed::Vector<double> delta_;
    ACCESSOR_READ_ONLY(delta)

    typename HighOrder<dim>::vector_type bounds_;

    vector_type r_;

    dealii::SparseMatrix<double> lij_matrix_; // FIXME
    std::array<dealii::SparseMatrix<double>, problem_dimension> pij_matrix_;

    vector_type temp_euler_;
    vector_type temp_ssprk_;

    /* Options: */

    bool use_ssprk_;
  };

} /* namespace grendel */

#endif /* TIME_STEP_H */
