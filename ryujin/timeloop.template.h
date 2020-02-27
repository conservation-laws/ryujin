#ifndef TIMELOOP_TEMPLATE_H
#define TIMELOOP_TEMPLATE_H

#include "checkpointing.h"
#include "timeloop.h"

#include <helper.h>
#include <indicator.h>
#include <limiter.h>
#include <riemann_solver.h>
#include <scope.h>

#include <deal.II/base/logstream.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#ifdef CALLGRIND
#include <valgrind/callgrind.h>
#endif

#include <fstream>
#include <iomanip>

using namespace dealii;
using namespace grendel;

namespace ryujin
{
  template <int dim, typename Number>
  TimeLoop<dim, Number>::TimeLoop(const MPI_Comm &mpi_comm)
      : ParameterAcceptor("A - TimeLoop")
      , mpi_communicator(mpi_comm)
      , discretization(mpi_communicator, "B - Discretization")
      , offline_data(mpi_communicator, discretization, "C - OfflineData")
      , initial_values("D - InitialValues")
      , time_step(mpi_communicator,
                  computing_timer,
                  offline_data,
                  initial_values,
                  "E - TimeStep")
      , postprocessor(mpi_communicator, offline_data, "F - Postprocessor")
      , mpi_rank(dealii::Utilities::MPI::this_mpi_process(mpi_communicator))
      , n_mpi_processes(
            dealii::Utilities::MPI::n_mpi_processes(mpi_communicator))
      , output_thread_active(0)
  {
    base_name = "cylinder";
    add_parameter("basename", base_name, "Base name for all output files");

    t_final = Number(5.);
    add_parameter("final time", t_final, "Final time");

    output_granularity = Number(0.01);
    add_parameter(
        "output granularity",
        output_granularity,
        "The output granularity specifies the time interval after which output "
        "routines are run. Further modified by \"*_multiplier\" options");

    enable_checkpointing = true;
    add_parameter(
        "enable checkpointing",
        enable_checkpointing,
        "Write out checkpoints to resume an interrupted computation "
        "at output granularity intervals. The frequency is determined by "
        "\"output granularity\" times \"output checkpoint multiplier\"");

    enable_output_full = true;
    add_parameter("enable output full",
                  enable_output_full,
                  "Write out full pvtu records. The frequency is determined by "
                  "\"output granularity\" times \"output full multiplier\"");

    enable_output_cutplanes = true;
    add_parameter(
        "enable output cutplanes",
        enable_output_cutplanes,
        "Write out cutplanes pvtu records. The frequency is determined by "
        "\"output granularity\" times \"output cutplanes multiplier\"");

    output_checkpoint_multiplier = 1;
    add_parameter("output checkpoint multiplier",
                  output_checkpoint_multiplier,
                  "Multiplicative modifier applied to \"output granularity\" "
                  "that determines the checkpointing granularity");

    output_full_multiplier = 1;
    add_parameter("output full multiplier",
                  output_full_multiplier,
                  "Multiplicative modifier applied to \"output granularity\" "
                  "that determines the full pvtu writeout granularity");

    output_cutplanes_multiplier = 1;
    add_parameter("output cutplanes multiplier",
                  output_cutplanes_multiplier,
                  "Multiplicative modifier applied to \"output granularity\" "
                  "that determines the cutplanes pvtu writeout granularity");

    enable_compute_error = false;
    add_parameter("enable compute error",
                  enable_compute_error,
                  "Flag to control whether we compute the Linfty Linf_norm of "
                  "the difference to an analytic solution. Implemented only "
                  "for certain initial state configurations.");

    resume = false;
    add_parameter("resume", resume, "Resume an interrupted computation");

    terminal_update_interval = 10;
    add_parameter("terminal update interval",
                  terminal_update_interval,
                  "number of cycles after which output statistics are "
                  "recomputed and printed on the terminal");
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::run()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeLoop<dim, Number>::run()" << std::endl;
#endif

    const bool write_output_files =
        enable_checkpointing || enable_output_full || enable_output_cutplanes;

    initialize();
    print_parameters(logfile);

    Number t = 0.;
    unsigned int output_cycle = 0;
    vector_type U;

    /* Prepare data structures: */

    {
      Scope scope(computing_timer, "initialize data structures");
      print_info("initializing data structures");

      discretization.prepare();
      offline_data.prepare();
      time_step.prepare();
      postprocessor.prepare();

      print_mpi_partition(logfile);

      const auto &partitioner = offline_data.partitioner();
      for (auto &it : U)
        it.reinit(partitioner);

      if (resume) {
        print_info("resuming interrupted computation");
        do_resume(base_name,
                  discretization.triangulation().locally_owned_subdomain(),
                  U,
                  t,
                  output_cycle);
      } else {
        print_info("interpolating initial values");
        U = initial_values.interpolate(offline_data);
      }
    }

    if (write_output_files) {
      output(U, base_name + "-solution", t, output_cycle);

      if (enable_compute_error) {
        const auto analytic = initial_values.interpolate(offline_data, t);
        output(analytic, base_name + "-analytic_solution", t, output_cycle);
      }
    }
    ++output_cycle;

    print_info("entering main loop");
    computing_timer["time loop"].start();

    /* Loop: */

    unsigned int cycle = 1;
    for (; t < t_final; ++cycle) {

#ifdef DEBUG_OUTPUT
      std::cout << "\n\n###   cycle = " << cycle << "   ###\n\n" << std::endl;
#endif

      /* Do a time step: */

      const auto tau = time_step.step(U, t);
      t += tau;

      if (t > output_cycle * output_granularity && write_output_files) {
        output(U,
               base_name + "-solution",
               t,
               output_cycle,
               /*checkpoint*/ true);

        if (enable_compute_error) {
          const auto analytic = initial_values.interpolate(offline_data, t);
          output(analytic, base_name + "-analytic_solution", t, output_cycle);
        }
        ++output_cycle;

        print_cycle_statistics(cycle, t, output_cycle, /*logfile*/ true);
      }

      if (cycle % terminal_update_interval == 0)
        print_cycle_statistics(cycle, t, output_cycle);
    } /* end of loop */

    --cycle; /* We have actually performed one cycle less. */

#ifdef CALLGRIND
    CALLGRIND_DUMP_STATS;
#endif

    /* Wait for output thread: */
    if (output_thread.joinable())
      output_thread.join();

    computing_timer["time loop"].stop();

    if (enable_compute_error) {
      /* Output final error: */
      compute_error(U, t);
    }

    /* Write final timing statistics to logfile: */
    print_cycle_statistics(cycle, t, output_cycle, /*final_time=*/ true);
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::initialize()
  {
    /* Read in parameters and initialize all objects: */

    if (mpi_rank == 0) {
      std::cout << "[Init] initiating flux capacitor" << std::endl;
      ParameterAcceptor::initialize("ryujin.prm");
    } else {
      ParameterAcceptor::initialize("ryujin.prm");
      return;
    }

    /* Print out parameters to a prm file: */

    std::ofstream output(base_name + "-parameter.prm");
    ParameterAcceptor::prm.print_parameters(output, ParameterHandler::Text);

    /* Attach log file: */
    logfile.open(base_name + ".log");
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::compute_error(
      const typename TimeLoop<dim, Number>::vector_type &U, const Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeLoop<dim, Number>::compute_error()" << std::endl;
#endif

    constexpr auto problem_dimension =
        ProblemDescription<dim, Number>::problem_dimension;

    /* Compute L_inf norm: */

    Vector<float> difference_per_cell(
        discretization.triangulation().n_active_cells());

    Number linf_norm = 0.;
    Number l1_norm = 0;
    Number l2_norm = 0;

    auto analytic = initial_values.interpolate(offline_data, t);

    for (unsigned int i = 0; i < problem_dimension; ++i) {
      auto &error = analytic[i];

      /* Compute norms of analytic solution: */

      const Number linf_norm_analytic =
          Utilities::MPI::max(error.linfty_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error,
                                        ZeroFunction<dim, Number>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L1_norm);

      const Number l1_norm_analytic =
          Utilities::MPI::sum(difference_per_cell.l1_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error,
                                        ZeroFunction<dim, Number>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L2_norm);

      const Number l2_norm_analytic = Number(std::sqrt(Utilities::MPI::sum(
          std::pow(difference_per_cell.l2_norm(), 2), mpi_communicator)));

      /* Compute norms of error: */

      error -= U[i];

      const Number linf_norm_error =
          Utilities::MPI::max(error.linfty_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error,
                                        ZeroFunction<dim, Number>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L1_norm);

      const Number l1_norm_error =
          Utilities::MPI::sum(difference_per_cell.l1_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error,
                                        ZeroFunction<dim, Number>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L2_norm);

      const Number l2_norm_error = Number(std::sqrt(Utilities::MPI::sum(
          std::pow(difference_per_cell.l2_norm(), 2), mpi_communicator)));

      linf_norm += linf_norm_error / linf_norm_analytic;
      l1_norm += l1_norm_error / l1_norm_analytic;
      l2_norm += l2_norm_error / l2_norm_analytic;
    }

    if (mpi_rank != 0)
      return;

    logfile << std::endl << "Computed errors:" << std::endl << std::endl;

    logfile << "Normalized consolidated Linf, L1, and L2 errors at "
            << "final time" << std::endl;
    logfile << "#dofs = " << offline_data.dof_handler().n_dofs() << std::endl;
    logfile << "t     = " << t << std::endl;
    logfile << "Linf  = " << linf_norm << std::endl;
    logfile << "L1    = " << l1_norm << std::endl;
    logfile << "L2    = " << l2_norm << std::endl;

    std::cout << "Normalized consolidated Linf, L1, and L2 errors at "
              << "final time" << std::endl;
    std::cout << "#dofs = " << offline_data.dof_handler().n_dofs() << std::endl;
    std::cout << "t     = " << t << std::endl;
    std::cout << "Linf  = " << linf_norm << std::endl;
    std::cout << "L1    = " << l1_norm << std::endl;
    std::cout << "L2    = " << l2_norm << std::endl;
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::output(
      const typename TimeLoop<dim, Number>::vector_type &U,
      const std::string &name,
      Number t,
      unsigned int cycle,
      bool checkpoint)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeLoop<dim, Number>::output(t = " << t
              << ", checkpoint = " << checkpoint << ")" << std::endl;
#endif

    /*
     * Offload output to a worker thread.
     *
     * We wait for a previous thread to finish before we schedule a new
     * one. This logic also serves as a mutex for the postprocessor class.
     */

    if (output_thread.joinable()) {
      Scope scope(computing_timer, "output stall");
      output_thread.join();
    }

    {
      Scope scope(computing_timer, "postprocessor");
      postprocessor.compute(U, time_step.alpha());
    }

    /* capture name, t, cycle, and checkpoint by value */
    const auto output_worker = [this, name, t, cycle, checkpoint]() {
      /* Flag thread as active: */
      output_thread_active = 1;
      Scope scope(computing_timer, "output write out");

      /* Checkpointing: */

      if (checkpoint && (cycle % output_checkpoint_multiplier == 0) &&
          enable_checkpointing) {
#ifdef DEBUG_OUTPUT
        std::cout << "        Checkpointing" << std::endl;
#endif
        do_checkpoint(base_name,
                      discretization.triangulation().locally_owned_subdomain(),
                      postprocessor.U(),
                      t,
                      cycle);
      }

      /* Data output: */
      postprocessor.write_out(name,
                              t,
                              cycle,
                              (cycle % output_full_multiplier == 0) &&
                                  enable_output_full,
                              (cycle % output_cutplanes_multiplier == 0) &&
                                  enable_output_cutplanes);

#ifdef DEBUG_OUTPUT
      std::cout << "        Commit output (cycle = " << cycle << ")"
                << std::endl;
#endif

      /* Flag thread as inactive: */
      output_thread_active = 0;
    };

    /* And spawn the thread: */

#ifdef DEBUG_OUTPUT
    std::cout << "        Schedule output (cycle = " << cycle << ")"
              << std::endl;
#endif

    if (!postprocessor.use_mpi_io()) {
#ifdef DEBUG_OUTPUT
      std::cout << "        Spawning worker thread" << std::endl;
#endif
      output_thread = std::move(std::thread(output_worker));
    } else {
      /* We unfortunately cannot run in a background thread if MPI IO is
       * enabled. Simply call the worker instead. */
      output_worker();
    }
  }


  /*
   * Output and logging related functions:
   */


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_parameters(std::ostream &stream)
  {
    if (mpi_rank != 0)
      return;

    /* Output commit and library informations: */

    /* clang-format off */
    stream << std::endl;
    stream << "###" << std::endl;
    stream << "#" << std::endl;
    stream << "# deal.II version " << std::setw(8) << DEAL_II_PACKAGE_VERSION
            << "  -  " << DEAL_II_GIT_REVISION << std::endl;
    stream << "# ryujin  version " << std::setw(8) << RYUJIN_VERSION
            << "  -  " << RYUJIN_GIT_REVISION << std::endl;
    stream << "#" << std::endl;
    stream << "###" << std::endl;

    /* Print compile time parameters: */

    stream << std::endl
           << std::endl << "Compile time parameters:" << std::endl << std::endl;

    stream << "DIM == " << dim << std::endl;
    stream << "NUMBER == " << typeid(Number).name() << std::endl;

#ifdef USE_SIMD
    stream << "SIMD width == " << VectorizedArray<Number>::n_array_elements << std::endl;
#else
    stream << "SIMD width == " << "(( disabled ))" << std::endl;
#endif

#ifdef USE_CUSTOM_POW
    stream << "serial pow == broadcasted pow(Vec4f)/pow(Vec2d)" << std::endl;
#else
    stream << "serial pow == std::pow"<< std::endl;
#endif

    stream << "Indicator<dim, Number>::indicators_ == ";
    switch (Indicator<dim, Number>::indicator_) {
    case Indicator<dim, Number>::Indicators::zero:
      stream << "Indicator<dim, Number>::Indicators::zero" << std::endl;
      break;
    case Indicator<dim, Number>::Indicators::one:
      stream << "Indicator<dim, Number>::Indicators::one" << std::endl;
      break;
    case Indicator<dim, Number>::Indicators::entropy_viscosity_commutator:
      stream << "Indicator<dim, Number>::Indicators::entropy_viscosity_commutator" << std::endl;
      break;
    case Indicator<dim, Number>::Indicators::smoothness_indicator:
      stream << "Indicator<dim, Number>::Indicators::smoothness_indicator" << std::endl;
    }

    stream << "Indicator<dim, Number>::smoothness_indicator_ == ";
    switch (Indicator<dim, Number>::smoothness_indicator_) {
    case Indicator<dim, Number>::SmoothnessIndicators::rho:
      stream << "Indicator<dim, Number>::SmoothnessIndicators::rho" << std::endl;
      break;
    case Indicator<dim, Number>::SmoothnessIndicators::internal_energy:
      stream << "Indicator<dim, Number>::SmoothnessIndicators::internal_energy" << std::endl;
      break;
    case Indicator<dim, Number>::SmoothnessIndicators::pressure:
      stream << "Indicator<dim, Number>::SmoothnessIndicators::pressure" << std::endl;
    }

    stream << "Indicator<dim, Number>::smoothness_indicator_alpha_0_ == "
            << Indicator<dim, Number>::smoothness_indicator_alpha_0_ << std::endl;

    stream << "Indicator<dim, Number>::smoothness_indicator_power_ == "
            << Indicator<dim, Number>::smoothness_indicator_power_ << std::endl;

    stream << "Indicator<dim, Number>::compute_second_variations_ == "
            << Indicator<dim, Number>::compute_second_variations_ << std::endl;

    stream << "Limiter<dim, Number>::limiter_ == ";
    switch (Limiter<dim, Number>::limiter_) {
    case Limiter<dim, Number>::Limiters::none:
      stream << "Limiter<dim, Number>::Limiters::none" << std::endl;
      break;
    case Limiter<dim, Number>::Limiters::rho:
      stream << "Limiter<dim, Number>::Limiters::rho" << std::endl;
      break;
    case Limiter<dim, Number>::Limiters::specific_entropy:
      stream << "Limiter<dim, Number>::Limiters::specific_entropy" << std::endl;
      break;
    case Limiter<dim, Number>::Limiters::entropy_inequality:
      stream << "Limiter<dim, Number>::Limiters::entropy_inequality" << std::endl;
      break;
    }

    stream << "grendel::newton_max_iter == "
            << grendel::newton_max_iter << std::endl;

    stream << "Limiter<dim, Number>::relax_bounds_ == "
            << Limiter<dim, Number>::relax_bounds_ << std::endl;

    stream << "Limiter<dim, Number>::relaxation_order_ == "
            << Limiter<dim, Number>::relaxation_order_ << std::endl;

    stream << "RiemannSolver<dim, Number>::newton_max_iter_ == "
            <<  RiemannSolver<dim, Number>::newton_max_iter_ << std::endl;

    stream << "RiemannSolver<dim, Number>::greedy_dij_ == "
            <<  RiemannSolver<dim, Number>::greedy_dij_ << std::endl;

    stream << "RiemannSolver<dim, Number>::greedy_threshold_ == "
            <<  RiemannSolver<dim, Number>::greedy_threshold_ << std::endl;

    stream << "RiemannSolver<dim, Number>::greedy_relax_bounds_ == "
            <<  RiemannSolver<dim, Number>::greedy_relax_bounds_ << std::endl;

    stream << "TimeStep<dim, Number>::order_ == ";
    switch (TimeStep<dim, Number>::order_) {
    case TimeStep<dim, Number>::Order::first_order:
      stream << "TimeStep<dim, Number>::Order::first_order" << std::endl;
      break;
    case TimeStep<dim, Number>::Order::second_order:
      stream << "TimeStep<dim, Number>::Order::second_order" << std::endl;
    }

    stream << "TimeStep<dim, Number>::time_step_order_ == ";
    switch (TimeStep<dim, Number>::time_step_order_) {
    case TimeStep<dim, Number>::TimeStepOrder::first_order:
      stream << "TimeStep<dim, Number>::TimeStepOrder::first_order" << std::endl;
      break;
    case TimeStep<dim, Number>::TimeStepOrder::second_order:
      stream << "TimeStep<dim, Number>::TimeStepOrder::second_order" << std::endl;
      break;
    case TimeStep<dim, Number>::TimeStepOrder::third_order:
      stream << "TimeStep<dim, Number>::TimeStepOrder::third_order" << std::endl;
      break;
    }

    stream << "TimeStep<dim, Number>::limiter_iter_ == "
            <<  TimeStep<dim, Number>::limiter_iter_ << std::endl;

    /* clang-format on */

    stream << std::endl;
    stream << std::endl << "Run time parameters:" << std::endl << std::endl;
    ParameterAcceptor::prm.print_parameters(
        stream, ParameterHandler::OutputStyle::ShortText);
    stream << std::endl;
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_mpi_partition(std::ostream &stream)
  {
    unsigned int dofs[4] = {offline_data.n_export_indices(),
                            offline_data.n_locally_internal(),
                            offline_data.n_locally_owned(),
                            offline_data.n_locally_relevant()};

    if (mpi_rank > 0) {
      MPI_Send(&dofs, 4, MPI_UNSIGNED, 0, 0, mpi_communicator);

    } else {

      stream << std::endl << "MPI partition:" << std::endl << std::endl;

      stream << "Number of MPI ranks: " << n_mpi_processes << std::endl;
      stream << "Number of threads:   " << MultithreadInfo::n_threads()
             << std::endl;

      /* Print out the DoF distribution: */

      const auto n_dofs = offline_data.dof_handler().n_dofs();

      stream << "Qdofs: " << n_dofs
             << " global DoFs, local DoF distribution:" << std::endl;


      for (unsigned int p = 0; p < n_mpi_processes; ++p) {
        stream << "    Rank " << p << std::flush;

        if (p != 0)
          MPI_Recv(&dofs,
                   4,
                   MPI_UNSIGNED,
                   p,
                   0,
                   mpi_communicator,
                   MPI_STATUS_IGNORE);

        stream << ":\t(exp) " << dofs[0] << ",\t(int) " << dofs[1]
               << ",\t(own) " << dofs[2] << ",\t(rel) " << dofs[3] << std::endl;
      } /* p */
    }   /* mpi_rank */
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_memory_statistics(std::ostream &stream)
  {
    std::ostringstream output;

    Utilities::System::MemoryStats stats;
    Utilities::System::get_memory_stats(stats);

    Utilities::MPI::MinMaxAvg data =
        Utilities::MPI::min_max_avg(stats.VmRSS / 1024., mpi_communicator);

    if (mpi_rank != 0)
      return;

    unsigned int n = dealii::Utilities::needed_digits(n_mpi_processes);

    output << std::endl << std::endl << "Memory:      [MiB]";
    output << std::setw(8) << data.min                        //
           << " [p" << std::setw(n) << data.min_index << "] " //
           << std::setw(8) << data.avg << " "                 //
           << std::setw(8) << data.max                        //
           << " [p" << std::setw(n) << data.max_index << "]"; //

    stream << output.str() << std::endl;
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_timers(std::ostream &stream)
  {
    std::vector<std::ostringstream> output(computing_timer.size());

    const auto equalize = [&]() {
      const auto ptr =
          std::max_element(output.begin(),
                           output.end(),
                           [](const auto &left, const auto &right) {
                             return left.str().length() < right.str().length();
                           });
      const auto length = ptr->str().length();
      for (auto &it : output)
        it << std::string(length - it.str().length() + 1, ' ');
    };

    const auto print_wall_time = [&](auto &timer, auto &stream) {
      const auto wall_time =
          Utilities::MPI::min_max_avg(timer.wall_time(), mpi_communicator);

      stream << std::setprecision(2) << std::fixed << std::setw(8)
             << wall_time.avg << "s [sk: " << std::setprecision(1)
             << std::setw(5) << std::fixed
             << 100. * (wall_time.max - wall_time.avg) / wall_time.avg << "%]";
      unsigned int n = dealii::Utilities::needed_digits(n_mpi_processes);
      stream << " [p" << std::setw(n) << wall_time.max_index << "]";
    };

    const auto cpu_time_statistics = Utilities::MPI::min_max_avg(
        computing_timer["time loop"].cpu_time(), mpi_communicator);
    const double total_cpu_time = cpu_time_statistics.sum;

    const auto print_cpu_time =
        [&](auto &timer, auto &stream, bool percentage) {
          const auto cpu_time =
              Utilities::MPI::min_max_avg(timer.cpu_time(), mpi_communicator);

          stream << std::setprecision(2) << std::fixed << std::setw(9)
                 << cpu_time.sum << "s ";

          if (percentage)
            stream << "(" << std::setprecision(1) << std::setw(4)
                   << 100. * cpu_time.sum / total_cpu_time << "%)";
          else
            stream << "       ";

          stream << " [sk: " << std::setprecision(1) << std::setw(5)
                 << std::fixed
                 << 100. * (cpu_time.max - cpu_time.avg) / cpu_time.avg << "%]";
        };

    auto jt = output.begin();
    for (auto &it : computing_timer)
      *jt++ << "  " << it.first;
    equalize();

    jt = output.begin();
    for (auto &it : computing_timer)
      print_wall_time(it.second, *jt++);
    equalize();

    jt = output.begin();
    for (auto &it : computing_timer)
      print_cpu_time(it.second, *jt++, it.first.find("time s") == 0);
    equalize();

    if (mpi_rank != 0)
      return;

    stream << std::endl << "Timer statistics:" << std::endl << std::endl;
    for (auto &it : output)
      stream << it.str() << std::endl;
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_throughput(unsigned int cycle,
                                               Number t,
                                               std::ostream &stream)
  {
    /* Print Jean-Luc and Martin metrics: */

    const auto wall_time_statistics = Utilities::MPI::min_max_avg(
        computing_timer["time loop"].wall_time(), mpi_communicator);
    const double wall_time = wall_time_statistics.max;

    const auto cpu_time_statistics = Utilities::MPI::min_max_avg(
        computing_timer["time loop"].cpu_time(), mpi_communicator);
    const double cpu_time = cpu_time_statistics.sum;

    const double wall_m_dofs_per_sec =
        ((double)cycle) * ((double)offline_data.dof_handler().n_dofs()) / 1.e6 /
        wall_time;

    const double cpu_m_dofs_per_sec =
        ((double)cycle) * ((double)offline_data.dof_handler().n_dofs()) / 1.e6 /
        cpu_time;

    std::ostringstream output;

    output << std::setprecision(4) << std::endl;
    output << "Throughput:  (CPU )  "                                    //
           << std::fixed << cpu_m_dofs_per_sec << " MQ/s  ("             //
           << std::scientific << 1. / cpu_m_dofs_per_sec * 1.e-6         //
           << " s/Qdof/cycle)" << std::endl;                             //
    output << "                     [cpu time skew: "                    //
           << std::setprecision(2) << std::scientific                    //
           << cpu_time_statistics.max - cpu_time_statistics.avg << "s (" //
           << std::setprecision(1) << std::setw(4) << std::setfill(' ')
           << std::fixed
           << 100. * (cpu_time_statistics.max - cpu_time_statistics.avg) /
                  cpu_time_statistics.avg
           << "%)]" << std::endl
           << std::endl;

    output << "             (WALL)  "                                      //
           << std::fixed << wall_m_dofs_per_sec << " MQ/s  ("              //
           << std::scientific << 1. / wall_m_dofs_per_sec * 1.e-6          //
           << " s/Qdof/cycle)  ("                                          //
           << std::fixed << ((double)cycle) / wall_time                    //
           << " cycles/s)  (avg dt = "                                     //
           << std::scientific << t / ((double)cycle)                       //
           << ")" << std::endl;                                            //
    output << "                     [ "                                    //
           << std::setprecision(0) << std::fixed << time_step.n_restarts() //
           << " rsts   (" << std::setprecision(2) << std::scientific
           << time_step.n_restarts() / ((double)cycle) << " rsts/cycle) ]"
           << std::endl
           << std::endl;

    output << "ETA:  " << std::fixed << std::setprecision(4)
           << ((t_final - t) / t * wall_time / 3600.) << " h";

    if (mpi_rank != 0)
      return;

    stream << output.str() << std::endl;
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_info(const std::string &header)
  {
    if (mpi_rank != 0)
      return;

    std::cout << "[Init] " << header << std::endl;
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_head(const std::string &header,
                                         const std::string &secondary,
                                         std::ostream &stream)
  {
    if (mpi_rank != 0)
      return;

    const auto header_size = header.size();
    const auto padded_header = std::string((34 - header_size) / 2, ' ') +
                               header +
                               std::string((35 - header_size) / 2, ' ');

    const auto secondary_size = secondary.size();
    const auto padded_secondary = std::string((34 - secondary_size) / 2, ' ') +
                                  secondary +
                                  std::string((35 - secondary_size) / 2, ' ');

    /* clang-format off */
    stream << "\n";
    stream << "    ####################################################\n";
    stream << "    #########                                  #########\n";
    stream << "    #########"     <<  padded_header   <<     "#########\n";
    stream << "    #########"     << padded_secondary <<     "#########\n";
    stream << "    #########                                  #########\n";
    stream << "    ####################################################\n";
    stream << std::endl;
    /* clang-format on */
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_cycle_statistics(unsigned int cycle,
                                                     Number t,
                                                     unsigned int output_cycle,
                                                     bool write_to_logfile,
                                                     bool final_time)
  {
    std::ostringstream output;

    unsigned int n_active_writebacks =
        Utilities::MPI::sum(output_thread_active, mpi_communicator);

    std::ostringstream primary;
    if (final_time) {
      primary << "FINAL  (cycle " << Utilities::int_to_string(cycle, 6) << ")";
    } else {
      primary << "Cycle  " << Utilities::int_to_string(cycle, 6) //
              << "  (" << std::fixed << std::setprecision(1)     //
              << t / t_final * 100 << "%)";
    }

    std::ostringstream secondary;
    secondary << "at time t = " << std::setprecision(8) << std::fixed << t;

    print_head(primary.str(), secondary.str(), output);

    output << "\n"
           << "Information: [" << base_name << "] with "
           << offline_data.dof_handler().n_dofs() << " Qdofs on "
           << n_mpi_processes << " ranks / " << MultithreadInfo::n_threads()
           << " threads\n"
           << "             Last output cycle " << output_cycle - 1
           << " at t = " << output_granularity * (output_cycle - 1)
           << std::endl;

    if (n_active_writebacks > 0)
      output << "             !!! " << n_active_writebacks
             << " ranks performing output !!!" << std::flush;

    print_memory_statistics(output);
    print_timers(output);
    print_throughput(cycle, t, output);

    if (mpi_rank == 0) {
#ifndef DEBUG_OUTPUT
      std::cout << "\033[2J\033[H";
#endif
      std::cout << output.str() << std::flush;
      if (write_to_logfile)
        logfile << "\n" << output.str() << std::flush;
    }
  }


} // namespace ryujin

#endif /* TIMELOOP_TEMPLATE_H */
