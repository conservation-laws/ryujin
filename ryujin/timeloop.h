#ifndef TIMELOOP_H
#define TIMELOOP_H

#include <discretization.h>
#include <riemann_solver.h>
#include <offline_data.h>
#include <time_step.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>

#include <sstream>

namespace ryujin
{

  template <int dim>
  class TimeLoop : public dealii::ParameterAcceptor
  {
  public:
    TimeLoop(const MPI_Comm &mpi_comm);
    virtual ~TimeLoop() final = default;

    virtual void run();

  private:
    /* Private methods for run(): */

    virtual void initialize();

    /* Data: */

    const MPI_Comm &mpi_communicator;
    std::ostringstream timer_output;
    dealii::TimerOutput computing_timer;

    std::string base_name;

    grendel::Discretization<dim> discretization;
    grendel::RiemannSolver<dim> riemann_solver;
    grendel::OfflineData<dim> offline_data;
    grendel::TimeStep<dim> time_step;

    std::unique_ptr<std::ofstream> filestream;

  };

} // namespace ryujin

#endif /* TIMELOOP_H */
