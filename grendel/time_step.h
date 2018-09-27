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

    void setup();

    std::tuple<vector_type, double> compute_step(const vector_type &U_old,
                                                 const double t_old);

  protected:

    const MPI_Comm &mpi_communicator_;
    dealii::TimerOutput &computing_timer_;

    dealii::SmartPointer<const grendel::OfflineData<dim>> offline_data_;
    A_RO(offline_data)

    dealii::SmartPointer<const grendel::ProblemDescription<dim>>
        problem_description_;
    A_RO(problem_description)

    dealii::SmartPointer<const grendel::RiemannSolver<dim>> riemann_solver_;
    A_RO(riemann_solver)

  private:
    dealii::Vector<rank2_type> f_i_;
    dealii::SparseMatrix<double> dij_matrix_;
  };

} /* namespace grendel */

#endif /* TIME_STEP_H */
