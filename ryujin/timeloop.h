#ifndef TIMELOOP_H
#define TIMELOOP_H

#include <discretization.h>
#include <offline_data.h>
#include <initial_values.h>
#include <postprocessor.h>
#include <time_step.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>

#include <sstream>

namespace ryujin
{

  template <int dim, typename Number = double>
  class TimeLoop : public dealii::ParameterAcceptor
  {
  public:
    using vector_type = typename grendel::TimeStep<dim, Number>::vector_type;

    TimeLoop(const MPI_Comm &mpi_comm);
    virtual ~TimeLoop() final = default;

    void run();

  private:
    /* Private methods for run(): */

    void initialize();

    vector_type interpolate_initial_values(Number t = 0);

    void compute_error(const vector_type &U, Number t);

    void output(const vector_type &U,
                const std::string &name,
                Number t,
                unsigned int cycle,
                bool checkpoint = false);

    void print_throughput(unsigned int cycle, Number t);

    /* Data: */

    const MPI_Comm &mpi_communicator;
    std::ostringstream timer_output;
    dealii::TimerOutput computing_timer;

    std::string base_name;
    Number t_final;
    Number output_granularity;

    bool enable_detailed_output;

    bool enable_checkpointing;
    bool resume;

    bool write_mesh;
    bool write_output_files;

    bool enable_compute_error;

    grendel::Discretization<dim> discretization;
    grendel::OfflineData<dim, Number> offline_data;
    grendel::InitialValues<dim, Number> initial_values;
    grendel::TimeStep<dim, Number> time_step;
    grendel::Postprocessor<dim, Number> postprocessor;

    std::unique_ptr<std::ofstream> filestream;

    /* Data for output management: */
    std::thread output_thread;
    vector_type output_vector;
    dealii::LinearAlgebra::distributed::Vector<Number> output_alpha;
  };

} // namespace ryujin

#endif /* TIMELOOP_H */
