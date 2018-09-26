#ifndef TIME_STEP_H
#define TIME_STEP_H

#include "boilerplate.h"
#include "offline_data.h"
#include "riemann_solver.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>

namespace grendel
{

  template <int dim>
  class TimeStep : public dealii::ParameterAcceptor
  {
  public:
    TimeStep(const MPI_Comm &mpi_communicator,
             dealii::TimerOutput &computing_timer,
             const grendel::OfflineData<dim> &offline_data,
             const grendel::RiemannSolver<dim> &riemann_solver,
             const std::string &subsection = "TimeStep");

    virtual ~TimeStep() final = default;

    void setup();

  protected:

    const MPI_Comm &mpi_communicator_;
    dealii::TimerOutput &computing_timer_;

    dealii::SmartPointer<const grendel::OfflineData<dim>> offline_data_;
    A_RO(offline_data)

    dealii::SmartPointer<const grendel::RiemannSolver<dim>> riemann_solver_;
    A_RO(riemann_solver)

  private:
    dealii::Vector<double> f_i_;
    dealii::SparseMatrix<double> dij_matrix_;
  };

} /* namespace grendel */

#endif /* TIME_STEP_H */
