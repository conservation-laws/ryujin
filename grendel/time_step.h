#ifndef TIME_STEP_H
#define TIME_STEP_H

#include "boilerplate.h"
#include "offline_data.h"
#include "problem_description.h"
#include "riemann_solver.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/vector.templates.h>
#include <deal.II/lac/la_parallel_vector.templates.h>

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
             const std::string &subsection = "TimeStep");

    virtual ~TimeStep() final = default;

    void prepare();

    /**
     * Given a reference to an previous state vector U_old compute a new
     * vector U_New by performing one explicit euler step. The function
     * return the chosen time step size tau and populates U_new by
     * reference.
     */
    double euler_step(vector_type &U_new, const vector_type &U_old);

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

  private:
    /* Scratch data: */
    std::vector<rank2_type> f_i_;
    dealii::SparseMatrix<double> dij_matrix_;
  };

} /* namespace grendel */

#endif /* TIME_STEP_H */
