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

#include <fstream>
#include <iomanip>


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
      , limiter(problem_description, "F - Limiter")
      , time_step(mpi_communicator,
                  computing_timer,
                  offline_data,
                  problem_description,
                  riemann_solver,
                  limiter,
                  "G - TimeStep")
      , schlieren_postprocessor(mpi_communicator,
                                computing_timer,
                                offline_data,
                                "H - SchlierenPostprocessor")
  {
    base_name = "test";
    add_parameter("basename", base_name, "Base name for all output files");

    t_final = 4.;
    add_parameter("final time", t_final, "Final time");

    output_granularity = 0.02;
    add_parameter(
        "output granularity", output_granularity, "time interval for output");

    enable_deallog_output = true;
    add_parameter("enable deallog output",
                  enable_deallog_output,
                  "Flag to control whether we output to deallog");

    enable_compute_error = false;
    add_parameter(
        "enable compute error",
        enable_compute_error,
        "Flag to control whether we compute the Linfty Linf_norm of the "
        "difference to an analytic solution. Implemented only for "
        "certain initial state configurations.");
  }


  template <int dim>
  void TimeLoop<dim>::run()
  {
    /*
     * Initialize deallog:
     */

    initialize();

    /*
     * Create distributed triangulation and output the triangulation to inp
     * files:
     */

    print_head("create triangulation");
    discretization.prepare();

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
    time_step.prepare();
    schlieren_postprocessor.prepare();

    /*
     * Interpolate initial values:
     */

    print_head("interpolate initial values");

    auto U = interpolate_initial_values();
    double t = 0.;
    double last_output = 0.;

    output(U, base_name + "-solution", t, 0);
    if (enable_compute_error) {
      output(U, base_name + "-analytic_solution", t, 0);
    }

    /*
     * Loop:
     */

    unsigned int output_cycle = 1;
    for (unsigned int cycle = 1; t < t_final; ++cycle) {
      std::ostringstream head;
      head << "Cycle  " << Utilities::int_to_string(cycle, 6)         //
           << "  ("                                                   //
           << std::fixed << std::setprecision(1) << t / t_final * 100 //
           << "%)";
      print_head(head.str());

      deallog << "        at time t="                    //
              << std::setprecision(8) << std::fixed << t //
              << std::endl;

      const auto tau = time_step.step(U);
      t += tau;

      if (t - last_output > output_granularity) {
        output(U, base_name + "-solution", t, output_cycle++);
        if (enable_compute_error) {
          const auto analytic = interpolate_initial_values(t);
          output(analytic, base_name + "-analytic_solution", t, output_cycle);
        }
        last_output = t;
      }
    } /* end of loop */

    /* Final output: */

    if (last_output < t_final) {
      output(U, base_name + "-solution", t, output_cycle);
      if (enable_compute_error) {
        const auto analytic = interpolate_initial_values(t);
        output(analytic, base_name + "-analytic_solution", t, output_cycle);
      }
    }

    computing_timer.print_summary();
    deallog << timer_output.str() << std::endl;

    /* Wait for output thread: */

    if (output_thread.joinable())
      output_thread.join();

    /* Output final error: */
    if (enable_compute_error) {
      compute_error(U, t);
    }

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

      if (!enable_deallog_output) {
        deallog.push("SILENT");
        deallog.depth_console(0);
        return;
      }

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


  template <int dim>
  typename TimeLoop<dim>::vector_type
  TimeLoop<dim>::interpolate_initial_values(double t)
  {
    deallog << "TimeLoop<dim>::interpolate_initial_values(t = " << t << ")"
            << std::endl;
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
      return problem_description.initial_state(p, t);
    };

    for (unsigned int i = 0; i < problem_dimension; ++i)
      VectorTools::interpolate(offline_data.dof_handler(),
                               to_function<dim, double>(callable, i),
                               U[i]);

    for (auto &it : U)
      it.update_ghost_values();

    return U;
  }


  template <int dim>
  void
  TimeLoop<dim>::compute_error(const typename TimeLoop<dim>::vector_type &U,
                               const double t)
  {
    deallog << "TimeLoop<dim>::compute_error()" << std::endl;
    TimerOutput::Scope timer(computing_timer, "time_loop - compute error");

    constexpr auto problem_dimension =
        ProblemDescription<dim>::problem_dimension;

    /*
     * Compute L_inf norm:
     */

    Vector<float> difference_per_cell(
        offline_data.discretization().triangulation().n_active_cells());

    double linf_norm = 0.;
    double l1_norm = 0;
    double l2_norm = 0;

    auto analytic = interpolate_initial_values(t);

    for (unsigned int i = 0; i < problem_dimension; ++i) {
      auto &error = analytic[i];

      /*
       * Compute norms of analytic solution:
       */

      const double linf_norm_analytic =
          Utilities::MPI::max(error.linfty_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error,
                                        ZeroFunction<dim, double>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L1_norm);

      const double l1_norm_analytic =
          Utilities::MPI::sum(difference_per_cell.l1_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error,
                                        ZeroFunction<dim, double>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L2_norm);

      const double l2_norm_analytic = std::sqrt(Utilities::MPI::sum(
          std::pow(difference_per_cell.l2_norm(), 2), mpi_communicator));

      /*
       * Compute norms of error:
       */

      error -= U[i];

      const double linf_norm_error =
          Utilities::MPI::max(error.linfty_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error,
                                        ZeroFunction<dim, double>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L1_norm);

      const double l1_norm_error =
          Utilities::MPI::sum(difference_per_cell.l1_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error,
                                        ZeroFunction<dim, double>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L2_norm);

      const double l2_norm_error = std::sqrt(Utilities::MPI::sum(
          std::pow(difference_per_cell.l2_norm(), 2), mpi_communicator));

      linf_norm += linf_norm_error / linf_norm_analytic;
      l1_norm += l1_norm_error / l1_norm_analytic;
      l2_norm += l2_norm_error / l2_norm_analytic;
    }

    deallog << "        Normalized consolidated Linf, L1, and L2 errors at "
            << "final time" << std::endl;
    deallog << "        #dofs = " << offline_data.dof_handler().n_dofs()
            << std::endl;
    deallog << "        t     = " << t << std::endl;
    deallog << "        Linf  = " << linf_norm << std::endl;
    deallog << "        L1    = " << l1_norm << std::endl;
    deallog << "        L2    = " << l2_norm << std::endl;
  }


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
     * We wait for a previous thread to finish before we schedule a new
     * one. This logic also serves as a mutex for output_vector and
     * schlieren_postprocessor.
     */

    deallog << "        Schedule output for cycle = " << cycle << std::endl;
    if (output_thread.joinable()) {
      TimerOutput::Scope timer(computing_timer, "time_loop - stalled output");
      output_thread.join();
    }

    /*
     * Copy the current state vector over to output_vector:
     */

    constexpr auto problem_dimension =
        ProblemDescription<dim>::problem_dimension;
    const auto &component_names = ProblemDescription<dim>::component_names;

    for (unsigned int i = 0; i < problem_dimension; ++i) {
      /* This also copies ghost elements: */
      output_vector[i] = U[i];
    }

    schlieren_postprocessor.compute_schlieren(output_vector);
    output_alpha = time_step.alpha();

    /* capture name, t, cycle by value */
    const auto output_worker = [this, name, t, cycle]() {

      const auto &dof_handler = offline_data.dof_handler();
      const auto &triangulation = discretization.triangulation();
      const auto &mapping = discretization.mapping();

      dealii::DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);

      for (unsigned int i = 0; i < problem_dimension; ++i)
        data_out.add_data_vector(output_vector[i], component_names[i]);

      data_out.add_data_vector(schlieren_postprocessor.schlieren(),
                               "schlieren_plot");

      output_alpha.update_ghost_values();
      data_out.add_data_vector(output_alpha, "alpha");

      data_out.build_patches(mapping,
                             discretization.finite_element().degree - 1);

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
     * And spawn the thread:
     */
    output_thread = std::move(std::thread(output_worker));
  }

} // namespace ryujin

#endif /* TIMELOOP_TEMPLATE_H */
