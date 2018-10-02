#ifndef TIMELOOP_TEMPLATE_H
#define TIMELOOP_TEMPLATE_H

#include "timeloop.h"

#include <helper.h>

#include <deal.II/base/logstream.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <iomanip>
#include <fstream>


using namespace dealii;
using namespace grendel;


namespace
{
  /**
   * A helper function to print formatted section headings.
   */
  void print_head(std::string header)
  {
    const auto size = header.size();

    deallog << std::endl;
    deallog << "    ####################################################"
            << std::endl;
    deallog << "    #########                                  #########"
            << std::endl;
    deallog << "    #########"                   //
            << std::string((34 - size) / 2, ' ') //
            << header                            //
            << std::string((35 - size) / 2, ' ') //
            << "#########"                       //
            << std::endl;
    deallog << "    #########                                  #########"
            << std::endl;
    deallog << "    ####################################################"
            << std::endl;
    deallog << std::endl;
  }
} // namespace


namespace ryujin
{
  template <int dim>
  TimeLoop<dim>::TimeLoop(const MPI_Comm &mpi_comm)
      : ParameterAcceptor("A - TimeLoop")
      , mpi_communicator(mpi_comm)
      , computing_timer(mpi_communicator,
                        timer_output,
                        TimerOutput::never,
                        TimerOutput::cpu_times)
      , discretization(mpi_communicator, computing_timer, "B - Discretization")
      , offline_data(mpi_communicator,
                     computing_timer,
                     discretization,
                     "C - OfflineData")
      , problem_description("D - ProblemDescription")
      , riemann_solver(problem_description, "E - RiemannSolver")
      , time_step(mpi_communicator,
                  computing_timer,
                  offline_data,
                  problem_description,
                  riemann_solver,
                  "F - TimeStep")
  {
    base_name = "test";
    add_parameter("basename", base_name, "Base name for all output files");

    t_final = 4.;
    add_parameter("final time", t_final, "Final time");

    output_granularity = 0.02;
    add_parameter(
        "output granularity", output_granularity, "time interval for output");
  }


  template <int dim>
  void TimeLoop<dim>::run()
  {
    initialize();

    print_head("create triangulation");
    discretization.create_triangulation();

    {
      deallog << "        output triangulation" << std::endl;
      std::ofstream output(
          base_name + "-triangulation-p" +
          std::to_string(Utilities::MPI::this_mpi_process(mpi_communicator)) +
          ".inp");
      GridOut().write_ucd(discretization.triangulation(), output);
    }

    /*
     * Prepare offline data:
     */

    print_head("compute offline data");
    offline_data.prepare();

    print_head("set up time step");
    time_step.setup();

    /*
     * Interpolate initial values and prepare U_new and U_output vectors:
     */

    print_head("interpolate initial values");

    auto U = interpolate_initial_values();
    double t = 0.;
    double last_output = 0.;

    output(U, base_name + "-solution", t, 0);

    vector_type U_new;
    for (auto &it : U_new)
      it.reinit(U[0]);

    /*
     * Loop:
     */

    for(unsigned int cycle = 1; t < t_final; ++cycle)
    {
      std::ostringstream head;
      head << "Cycle  " << Utilities::int_to_string(cycle, 6)         //
           << "  ("                                                   //
           << std::fixed << std::setprecision(1) << t / t_final * 100 //
           << "%)";
      print_head(head.str());

      deallog << "        at time t="                    //
              << std::setprecision(4) << std::fixed << t //
              << std::endl;

      const auto tau = time_step.euler_step(U_new, U);
      t += tau;
      U.swap(U_new);

      if (t - last_output > output_granularity) {
        output(U, base_name + "-solution", t, cycle);
        last_output = t;
      }
    }

    computing_timer.print_summary();
    deallog << timer_output.str() << std::endl;

    /* Wait for output thread: */

    if (output_thread.joinable())
      output_thread.join();

    /* Detach deallog: */

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
      deallog.pop();
      deallog.detach();
    }
  }


  /**
   * Set up deallog output, read in parameters and initialize all objects.
   */
  template <int dim>
  void TimeLoop<dim>::initialize()
  {
    /* Read in parameters and initialize all objects: */

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
      deallog.pop();

      deallog << "[Init] Initiating Flux Capacitor... [ OK ]" << std::endl;
      deallog << "[Init] Bringing Warp Core online... [ OK ]" << std::endl;

      deallog << "[Init] Reading parameters and allocating objects... "
              << std::flush;

      ParameterAcceptor::initialize("ryujin.prm");

      deallog << "[ OK ]" << std::endl;

    } else {

      ParameterAcceptor::initialize("ryujin.prm");
      return;
    }

    /* Print out parameters to a prm file: */

    std::ofstream output(base_name + "-parameter.prm");
    ParameterAcceptor::prm.print_parameters(output, ParameterHandler::Text);

    /* Prepare deallog: */

    deallog.push(DEAL_II_GIT_SHORTREV "+" RYUJIN_GIT_SHORTREV);
#ifdef DEBUG
    deallog.depth_console(5);
    deallog.push("DEBUG");
#else
    deallog.depth_console(4);
#endif
    deallog.push(base_name);

    /* Prepare and attach logfile: */

    filestream.reset(new std::ofstream(base_name + "-deallog.log"));
    deallog.attach(*filestream);

    /* Output commit and library informations: */

    /* clang-format off */
    deallog << "###" << std::endl;
    deallog << "#" << std::endl;
    deallog << "# deal.II version " << std::setw(8) << DEAL_II_PACKAGE_VERSION
            << "  -  " << DEAL_II_GIT_REVISION << std::endl;
    deallog << "# ryujin  version " << std::setw(8) << RYUJIN_VERSION
            << "  -  " << RYUJIN_GIT_REVISION << std::endl;
    deallog << "#" << std::endl;
    deallog << "###" << std::endl;
    /* clang-format on */

    /* Print out parameters to deallog as well: */

    deallog << "TimeLoop<dim>::run()" << std::endl;
    ParameterAcceptor::prm.log_parameters(deallog);
  }


  /**
   * Return a vector populated with initial values:
   */
  template <int dim>
  typename TimeLoop<dim>::vector_type
  TimeLoop<dim>::interpolate_initial_values()
  {
    deallog << "TimeLoop<dim>::interpolate_initial_values()" << std::endl;
    TimerOutput::Scope timer(computing_timer,
                             "time_loop - setup scratch space");

    vector_type U;

    const auto &locally_owned = offline_data.locally_owned();
    const auto &locally_relevant = offline_data.locally_relevant();
    U[0].reinit(locally_owned, locally_relevant, mpi_communicator);
    for (auto &it : U)
      it.reinit(U[0]);

    constexpr auto problem_dimension =
        ProblemDescription<dim>::problem_dimension;
    const auto callable = [&](const auto &p) {
      return problem_description.initial_state(p);
    };
    for (unsigned int i = 0; i < problem_dimension; ++i)
      VectorTools::interpolate(offline_data.dof_handler(),
                               to_function<dim, double>(callable, i),
                               U[i]);

    for (auto &it : U)
      it.update_ghost_values();

    return U;
  }


  /**
   * FIXME: Description
   */
  template <int dim>
  void TimeLoop<dim>::output(const typename TimeLoop<dim>::vector_type &U,
                             const std::string &name,
                             double t,
                             unsigned int cycle)
  {
    deallog << "TimeLoop<dim>::output()" << std::endl;

    /*
     * Offload output to a worker thread.
     *
     */

    /* capture U, name, t, cycle by value */
    const auto output_worker = [this, U, name, t, cycle]() {
      // FIXME: account for elapsed time in this thread as well.
      // TimerOutput::Scope timer(computing_timer, "time_loop - output");

      constexpr auto problem_dimension =
          ProblemDescription<dim>::problem_dimension;

      const auto &dof_handler = offline_data.dof_handler();
      const auto &triangulation = discretization.triangulation();
      const auto &mapping = discretization.mapping();

      // FIXME
      std::vector<std::string> component_names = {"rho", "m_1", "m_2", "E"};

      dealii::DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);

      for (unsigned int i = 0; i < problem_dimension; ++i)
        data_out.add_data_vector(U[i], component_names[i]);

      data_out.build_patches(mapping);

      DataOutBase::VtkFlags flags(
          t, cycle, true, DataOutBase::VtkFlags::best_speed);
      data_out.set_flags(flags);

      const auto filename = [&](const unsigned int i) -> std::string {
        const auto seq = dealii::Utilities::int_to_string(i, 4);
        return name + "-" + Utilities::int_to_string(cycle, 6) + "-" + seq +
               ".vtu";
      };

      /* Write out local vtu: */
      const unsigned int i = triangulation.locally_owned_subdomain();
      std::ofstream output(filename(i));
      data_out.write_vtu(output);

      if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        /* Write out pvtu control file: */

        const unsigned int n_mpi_processes =
            dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
        std::vector<std::string> filenames;
        for (unsigned int i = 0; i < n_mpi_processes; ++i)
          filenames.push_back(filename(i));

        std::ofstream output(name + "-" + Utilities::int_to_string(cycle, 6) +
                             ".pvtu");
        data_out.write_pvtu_record(output, filenames);
      }

      /*
       * Release output mutex:
       */
      deallog << "        Commit output for cycle = " << cycle << std::endl;
    };

    /*
     * Spawn thread. We wait for a previous thread to finish before we
     * schedule a new one.
     */

    deallog << "        Schedule output for cycle = " << cycle << std::endl;
    if (output_thread.joinable()) {
      TimerOutput::Scope timer(computing_timer, "time_loop - stalled output");
      output_thread.join();
    }
    output_thread = std::move(std::thread (output_worker));
  }

} // namespace ryujin

#endif /* TIMELOOP_TEMPLATE_H */
