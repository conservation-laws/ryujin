#ifndef TIMELOOP_H
#define TIMELOOP_H

#include <discretization.h>
#include <high_order.h>
#include <offline_data.h>
#include <problem_description.h>
#include <riemann_solver.h>
#include <schlieren_postprocessor.h>
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
    using vector_type = typename grendel::TimeStep<dim>::vector_type;

    TimeLoop(const MPI_Comm &mpi_comm);
    virtual ~TimeLoop() final = default;

    void run();

  private:
    /* Private methods for run(): */

    void initialize();

    vector_type interpolate_initial_values();

    double compute_error(const vector_type &U, double t);

    void output(const vector_type &U,
                const std::string &name,
                double t,
                unsigned int cycle);

    /* Data: */

    const MPI_Comm &mpi_communicator;
    std::ostringstream timer_output;
    dealii::TimerOutput computing_timer;

    std::string base_name;
    double t_final;
    double output_granularity;
    bool enable_deallog_output;
    bool enable_compute_error;

    grendel::Discretization<dim> discretization;
    grendel::OfflineData<dim> offline_data;
    grendel::ProblemDescription<dim> problem_description;
    grendel::RiemannSolver<dim> riemann_solver;
    grendel::HighOrder<dim> high_order;
    grendel::TimeStep<dim> time_step;
    grendel::SchlierenPostprocessor<dim> schlieren_postprocessor;

    std::unique_ptr<std::ofstream> filestream;

    /* Data for output management: */
    std::thread output_thread;
    vector_type output_vector;
    dealii::LinearAlgebra::distributed::Vector<double> output_alpha;
  };

} // namespace ryujin

#endif /* TIMELOOP_H */
