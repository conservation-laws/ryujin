//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef TIME_LOOP_TEMPLATE_H
#define TIME_LOOP_TEMPLATE_H

#include "checkpointing.h"
#include "indicator.h"
#include "limiter.h"
#include "riemann_solver.h"
#include "scope.h"
#include "time_loop.h"

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

namespace ryujin
{
  template <int dim, typename Number>
  TimeLoop<dim, Number>::TimeLoop(const MPI_Comm &mpi_comm)
      : ParameterAcceptor("/A - TimeLoop")
      , mpi_communicator(mpi_comm)
      , problem_description("/B - ProblemDescription")
      , discretization(mpi_communicator, "/C - Discretization")
      , offline_data(mpi_communicator, discretization, "/D - OfflineData")
      , initial_values(problem_description, "/E - InitialValues")
      , euler_module(mpi_communicator,
                     computing_timer,
                     offline_data,
                     problem_description,
                     initial_values,
                     "/F - EulerModule")
      , dissipation_module(mpi_communicator,
                           computing_timer,
                           offline_data,
                           problem_description,
                           initial_values,
                           "/G - DissipationModule")
      , vtu_output(mpi_communicator, offline_data, "/H - VTUOutput")
      , quantities(mpi_communicator,
                   problem_description,
                   offline_data,
                   "/I - Quantities")
      , mpi_rank(dealii::Utilities::MPI::this_mpi_process(mpi_communicator))
      , n_mpi_processes(
            dealii::Utilities::MPI::n_mpi_processes(mpi_communicator))
  {
    base_name = "cylinder";
    add_parameter("basename", base_name, "Base name for all output files");

    t_initial = Number(0.);

    t_final = Number(5.);
    add_parameter("final time", t_final, "Final time");

    output_granularity = Number(0.01);
    add_parameter(
        "output granularity",
        output_granularity,
        "The output granularity specifies the time interval after which output "
        "routines are run. Further modified by \"*_multiplier\" options");

    enable_checkpointing = false;
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

    enable_compute_error = false;
    add_parameter("enable compute error",
                  enable_compute_error,
                  "Flag to control whether we compute the Linfty Linf_norm of "
                  "the difference to an analytic solution. Implemented only "
                  "for certain initial state configurations.");

    enable_compute_quantities = false;
    add_parameter(
        "enable compute quantities",
        enable_compute_quantities,
        "Flag to control whether we compute quantities of interest. The "
        "frequency how often quantities are logged is determined by \"output "
        "granularity\" times \"output quantities multiplier\"");

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

    output_quantities_multiplier = 1;
    add_parameter(
        "output quantities multiplier",
        output_quantities_multiplier,
        "Multiplicative modifier applied to \"output granularity\" that "
        "determines the writeout granularity for quantities of interest");

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

    AssertThrow(!enable_checkpointing || !enable_compute_error,
                ExcNotImplemented());

    const bool write_output_files =
        enable_checkpointing || enable_output_full || enable_output_cutplanes;

    /* Attach log file: */
    if (mpi_rank == 0)
      logfile.open(base_name + ".log");

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
      euler_module.prepare();
      dissipation_module.prepare();
      vtu_output.prepare();
      quantities.prepare(base_name + "-quantities.log");

      print_mpi_partition(logfile);

      U.reinit(offline_data.vector_partitioner());

      if (resume) {
        print_info("resuming interrupted computation");
        const auto id =
            discretization.triangulation().locally_owned_subdomain();
        do_resume(base_name, id, U, t, output_cycle);
        t_initial = t;
      } else {
        print_info("interpolating initial values");
        U = initial_values.interpolate(offline_data);
#ifdef DEBUG
        /* Poison constrained degrees of freedom: */
        const unsigned int n_relevant = offline_data.n_locally_relevant();
        const auto &partitioner = offline_data.scalar_partitioner();
        for (unsigned int i = 0; i < n_relevant; ++i) {
          if (offline_data.affine_constraints().is_constrained(
                  partitioner->local_to_global(i)))
            U.write_tensor(dealii::Tensor<1, dim + 2, Number>() *
                               std::numeric_limits<Number>::signaling_NaN(),
                           i);
        }
#endif
      }
    }

    if (write_output_files) {
      output(U, base_name + "-solution", t, output_cycle);
      if (enable_compute_error) {
        const auto analytic = initial_values.interpolate(offline_data, t);
        output(analytic, base_name + "-analytic_solution", t, output_cycle);
      }
    }
    if (enable_compute_quantities)
      quantities.compute(U, t);
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

      if (problem_description.description() == "Euler") {

        /* Pure hyperbolic update: */
        const auto tau = euler_module.step(U, t);
        t += tau;

      } else if (problem_description.description() == "Navier Stokes") {

        /* Strang's splitting: */
        const auto tau = euler_module.step(U, t);
        dissipation_module.step(U, t, 2. * tau);
        euler_module.step(U, t + tau, tau);
        t += 2. * tau;

      } else {

        AssertThrow(false, ExcMessage("Unknown problem description"));
      }

      if (t > output_cycle * output_granularity) {
        if (write_output_files) {
          output(U, base_name + "-solution", t, output_cycle);
          if (enable_compute_error) {
            const auto analytic = initial_values.interpolate(offline_data, t);
            output(analytic, base_name + "-analytic_solution", t, output_cycle);
          }
        }
        if (enable_compute_quantities &&
            (output_cycle % output_quantities_multiplier == 0))
          quantities.compute(U, t);

        ++output_cycle;

        print_cycle_statistics(cycle, t, output_cycle, /*logfile*/ true);
      }

      if (cycle % terminal_update_interval == 0)
        print_cycle_statistics(cycle, t, output_cycle);
    } /* end of loop */

    /* Wait for output thread: */
    vtu_output.wait();

    /* We have actually performed one cycle less. */
    --cycle;

    computing_timer["time loop"].stop();

    /* Write final timing statistics to logfile: */
    print_cycle_statistics(cycle, t, output_cycle, /*final_time=*/true);

    if (enable_compute_error) {
      /* Output final error: */
      compute_error(U, t);
    }

#ifdef CALLGRIND
    CALLGRIND_DUMP_STATS;
#endif
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::compute_error(
      const typename TimeLoop<dim, Number>::vector_type &U, const Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeLoop<dim, Number>::compute_error()" << std::endl;
#endif

    constexpr auto problem_dimension =
        ProblemDescription::problem_dimension<dim>;

    /* Compute L_inf norm: */

    Vector<float> difference_per_cell(
        discretization.triangulation().n_active_cells());

    Number linf_norm = 0.;
    Number l1_norm = 0;
    Number l2_norm = 0;

    const auto analytic = initial_values.interpolate(offline_data, t);

    scalar_type analytic_component;
    scalar_type error_component;
    analytic_component.reinit(offline_data.scalar_partitioner());
    error_component.reinit(offline_data.scalar_partitioner());

    for (unsigned int i = 0; i < problem_dimension; ++i) {

      /* Compute norms of analytic solution: */

      analytic.extract_component(analytic_component, i);

      const Number linf_norm_analytic = Utilities::MPI::max(
          analytic_component.linfty_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        analytic_component,
                                        Functions::ZeroFunction<dim, Number>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L1_norm);

      const Number l1_norm_analytic =
          Utilities::MPI::sum(difference_per_cell.l1_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        analytic_component,
                                        Functions::ZeroFunction<dim, Number>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L2_norm);

      const Number l2_norm_analytic = Number(std::sqrt(Utilities::MPI::sum(
          std::pow(difference_per_cell.l2_norm(), 2), mpi_communicator)));

      /* Compute norms of error: */

      U.extract_component(error_component, i);
      /* Populate constrained dofs due to periodicity: */
      offline_data.affine_constraints().distribute(error_component);
      error_component.update_ghost_values();
      error_component -= analytic_component;

      const Number linf_norm_error =
          Utilities::MPI::max(error_component.linfty_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error_component,
                                        Functions::ZeroFunction<dim, Number>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L1_norm);

      const Number l1_norm_error =
          Utilities::MPI::sum(difference_per_cell.l1_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error_component,
                                        Functions::ZeroFunction<dim, Number>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L2_norm);

      const Number l2_norm_error = Number(std::sqrt(Utilities::MPI::sum(
          std::pow(difference_per_cell.l2_norm(), 2), mpi_communicator)));

      if (linf_norm_analytic >= 1.0e-10)
        linf_norm += linf_norm_error / linf_norm_analytic;
      if (l1_norm_analytic >= 1.0e-10)
        l1_norm += l1_norm_error / l1_norm_analytic;
      if (l2_norm_analytic >= 1.0e-10)
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
      unsigned int cycle)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeLoop<dim, Number>::output(t = " << t << ")" << std::endl;
#endif

    const bool do_full_output =
        (cycle % output_full_multiplier == 0) && enable_output_full;
    const bool do_cutplanes =
        (cycle % output_cutplanes_multiplier == 0) && enable_output_cutplanes;
    const bool do_checkpointing =
        (cycle % output_checkpoint_multiplier == 0) && enable_checkpointing;

    /* There is nothing to do: */
    if (!(do_full_output || do_cutplanes || do_checkpointing))
      return;

    /* Wait for a previous thread to finish before scheduling a new one: */
    {
      Scope scope(computing_timer, "output stall");
      print_info("waiting for previous output cycle to finish");

      vtu_output.wait();
    }

    /* Data output: */
    if (do_full_output || do_cutplanes) {
      Scope scope(computing_timer, "vtu output");
      print_info("scheduling output");

      vtu_output.schedule_output(U,
                                 euler_module.alpha(),
                                 name,
                                 t,
                                 cycle,
                                 do_full_output,
                                 do_cutplanes);
    }

    /* Checkpointing: */
    if (do_checkpointing) {
      Scope scope(computing_timer, "checkpointing");
      print_info("scheduling checkpointing");

      const auto id = discretization.triangulation().locally_owned_subdomain();
      do_checkpoint(base_name, id, U, t, cycle);
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
    stream << "SIMD width == " << VectorizedArray<Number>::size() << std::endl;
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

    stream << "Indicator<dim, Number>::compute_second_variations_ == "
            << Indicator<dim, Number>::compute_second_variations_ << std::endl;

    stream << "Indicator<dim, Number>::evc_entropy_ == ";
    switch (Indicator<dim, Number>::evc_entropy_) {
    case Indicator<dim, Number>::Entropy::mathematical:
      stream << "Indicator<dim, Number>::Entropy::mathematical" << std::endl;
      break;
    case Indicator<dim, Number>::Entropy::harten:
      stream << "Indicator<dim, Number>::Entropy::harten" << std::endl;
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

    stream << "ryujin::newton_max_iter == "
           << ryujin::newton_max_iter << std::endl;

    stream << "Limiter<dim, Number>::relax_bounds_ == "
           << Limiter<dim, Number>::relax_bounds_ << std::endl;

    stream << "Limiter<dim, Number>::relaxation_order_ == "
           << Limiter<dim, Number>::relaxation_order_ << std::endl;

    stream << "ProblemDescription::equation_of_state_ == ";
    switch (ProblemDescription::equation_of_state_) {
    case ProblemDescription::EquationOfState::ideal_gas:
      stream << "ProblemDescription::EquationOfState::ideal_gas" << std::endl;
      break;
    case ProblemDescription::EquationOfState::van_der_waals:
      stream << "ProblemDescription::EquationOfState::van_der_waals" << std::endl;
      break;
    }

    stream << "RiemannSolver<dim, Number>::newton_max_iter_ == "
           <<  RiemannSolver<dim, Number>::newton_max_iter_ << std::endl;

    /* clang-format on */

    stream << std::endl;
    stream << std::endl << "Run time parameters:" << std::endl << std::endl;
    ParameterAcceptor::prm.print_parameters(
        stream, ParameterHandler::OutputStyle::ShortText);
    stream << std::endl;

    /* Also print out parameters to a prm file: */

    std::ofstream output(base_name + "-parameter.prm");
    ParameterAcceptor::prm.print_parameters(output, ParameterHandler::Text);
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
             << 100. * (wall_time.min - wall_time.avg) / wall_time.avg << "%/"
             << std::setw(4) << std::fixed
             << 100. * (wall_time.max - wall_time.avg) / wall_time.avg << "%]";
      unsigned int n = dealii::Utilities::needed_digits(n_mpi_processes);
      stream << " [p" << std::setw(n) << wall_time.min_index << "/"
             << wall_time.max_index << "]";
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

    output << "             (WALL)  "                                         //
           << std::fixed << wall_m_dofs_per_sec << " MQ/s  ("                 //
           << std::scientific << 1. / wall_m_dofs_per_sec * 1.e-6             //
           << " s/Qdof/cycle)  ("                                             //
           << std::fixed << ((double)cycle) / wall_time                       //
           << " cycles/s)  (avg dt = "                                        //
           << std::scientific << t / ((double)cycle)                          //
           << ")" << std::endl;                                               //
    output << "                     [ "                                       //
           << std::setprecision(0) << std::fixed << euler_module.n_restarts() //
           << " rsts   (" << std::setprecision(2) << std::scientific
           << euler_module.n_restarts() / ((double)cycle) << " rsts/cycle) ]";
    output << "[ "                                                        //
           << std::setprecision(2) << std::fixed                          //
           << dissipation_module.n_iterations_velocity() << " CG vel -- " //
           << dissipation_module.n_iterations_internal_energy()           //
           << " CG int ]" << std::endl
           << std::endl;

    /* and print an ETA */
    unsigned int eta =
        static_cast<unsigned int>((t_final - t) / (t - t_initial) * wall_time);

    output << "ETA:  ";

    const unsigned int days = eta / (24 * 3600);
    if (days > 0) {
      output << days << " d  ";
      eta %= 24 * 3600;
    }

    const unsigned int hours = eta / 3600;
    if (hours > 0) {
      output << hours << " h  ";
      eta %= 3600;
    }

    const unsigned int minutes = eta / 60;
    output << minutes << " min" << std::endl;

    if (mpi_rank != 0)
      return;

    stream << output.str() << std::endl;
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_info(const std::string &header)
  {
    if (mpi_rank != 0)
      return;

    std::cout << "[INFO] " << header << std::endl;
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

    unsigned int n_active_writebacks = Utilities::MPI::sum<unsigned int>(
        vtu_output.is_active(), mpi_communicator);

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

#endif /* TIME_LOOP_TEMPLATE_H */
