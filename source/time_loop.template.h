//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "checkpointing.h"
#include "introspection.h"
#include "scope.h"
#include "solution_transfer.h"
#include "time_loop.h"

#include <deal.II/base/logstream.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include <fstream>
#include <iomanip>

using namespace dealii;

namespace ryujin
{
  template <int dim, typename Number>
  TimeLoop<dim, Number>::TimeLoop(const MPI_Comm &mpi_comm)
      : ParameterAcceptor("/A - TimeLoop")
      , mpi_communicator_(mpi_comm)
      , hyperbolic_system_("/B - HyperbolicSystem")
      , discretization_(mpi_communicator_, "/C - Discretization")
      , offline_data_(mpi_communicator_, discretization_, "/D - OfflineData")
      , initial_values_(hyperbolic_system_, offline_data_, "/E - InitialValues")
      , hyperbolic_module_(mpi_communicator_,
                           computing_timer_,
                           offline_data_,
                           hyperbolic_system_,
                           initial_values_,
                           "/F - HyperbolicModule")
      , time_integrator_(mpi_communicator_,
                         computing_timer_,
                         offline_data_,
                         hyperbolic_module_,
                         "/H - TimeIntegrator")
      , postprocessor_(mpi_communicator_,
                       hyperbolic_system_,
                       offline_data_,
                       "/I - VTUOutput/Postprocessor")
      , vtu_output_(mpi_communicator_,
                    offline_data_,
                    hyperbolic_module_,
                    postprocessor_,
                    "/I - VTUOutput")
      , quantities_(mpi_communicator_,
                    hyperbolic_system_,
                    offline_data_,
                    "/J - Quantities")
      , mpi_rank_(dealii::Utilities::MPI::this_mpi_process(mpi_communicator_))
      , n_mpi_processes_(
            dealii::Utilities::MPI::n_mpi_processes(mpi_communicator_))
  {
    base_name_ = "cylinder";
    add_parameter("basename", base_name_, "Base name for all output files");

    t_initial_ = Number(0.);

    t_final_ = Number(5.);
    add_parameter("final time", t_final_, "Final time");

    add_parameter("refinement timepoints",
                  t_refinements_,
                  "List of points in (simulation) time at which the mesh will "
                  "be globally refined");

    output_granularity_ = Number(0.01);
    add_parameter(
        "output granularity",
        output_granularity_,
        "The output granularity specifies the time interval after which output "
        "routines are run. Further modified by \"*_multiplier\" options");

    enable_checkpointing_ = false;
    add_parameter(
        "enable checkpointing",
        enable_checkpointing_,
        "Write out checkpoints to resume an interrupted computation "
        "at output granularity intervals. The frequency is determined by "
        "\"output granularity\" times \"output checkpoint multiplier\"");

    enable_output_full_ = false;
    add_parameter("enable output full",
                  enable_output_full_,
                  "Write out full pvtu records. The frequency is determined by "
                  "\"output granularity\" times \"output full multiplier\"");

    enable_output_levelsets_ = false;
    add_parameter(
        "enable output levelsets",
        enable_output_levelsets_,
        "Write out levelsets pvtu records. The frequency is determined by "
        "\"output granularity\" times \"output levelsets multiplier\"");

    enable_compute_error_ = false;
    add_parameter("enable compute error",
                  enable_compute_error_,
                  "Flag to control whether we compute the Linfty Linf_norm of "
                  "the difference to an analytic solution. Implemented only "
                  "for certain initial state configurations.");

    enable_compute_quantities_ = false;
    add_parameter(
        "enable compute quantities",
        enable_compute_quantities_,
        "Flag to control whether we compute quantities of interest. The "
        "frequency how often quantities are logged is determined by \"output "
        "granularity\" times \"output quantities multiplier\"");

    output_checkpoint_multiplier_ = 1;
    add_parameter("output checkpoint multiplier",
                  output_checkpoint_multiplier_,
                  "Multiplicative modifier applied to \"output granularity\" "
                  "that determines the checkpointing granularity");

    output_full_multiplier_ = 1;
    add_parameter("output full multiplier",
                  output_full_multiplier_,
                  "Multiplicative modifier applied to \"output granularity\" "
                  "that determines the full pvtu writeout granularity");

    output_levelsets_multiplier_ = 1;
    add_parameter("output levelsets multiplier",
                  output_levelsets_multiplier_,
                  "Multiplicative modifier applied to \"output granularity\" "
                  "that determines the levelsets pvtu writeout granularity");

    output_quantities_multiplier_ = 1;
    add_parameter(
        "output quantities multiplier",
        output_quantities_multiplier_,
        "Multiplicative modifier applied to \"output granularity\" that "
        "determines the writeout granularity for quantities of interest");

    resume_ = false;
    add_parameter("resume", resume_, "Resume an interrupted computation");

    terminal_update_interval_ = 10;
    add_parameter("terminal update interval",
                  terminal_update_interval_,
                  "number of cycles after which output statistics are "
                  "recomputed and printed on the terminal");
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::run()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeLoop<dim, Number>::run()" << std::endl;
#endif

    const bool write_output_files = enable_checkpointing_ ||
                                    enable_output_full_ ||
                                    enable_output_levelsets_;

    /* Attach log file: */
    if (mpi_rank_ == 0)
      logfile_.open(base_name_ + ".log");

    print_parameters(logfile_);

    Number t = 0.;
    unsigned int output_cycle = 0;
    vector_type U;

    /* Prepare data structures: */

    const auto prepare_compute_kernels = [&]() {
      offline_data_.prepare(HyperbolicSystem::problem_dimension<dim>);
      hyperbolic_module_.prepare();
      time_integrator_.prepare();
      postprocessor_.prepare();
      vtu_output_.prepare();
      /* We skip the first output cycle for quantities: */
      quantities_.prepare(base_name_, output_cycle == 0 ? 1 : output_cycle);
      print_mpi_partition(logfile_);
    };

    {
      Scope scope(computing_timer_, "(re)initialize data structures");
      print_info("initializing data structures");

      if (resume_) {
        print_info("resuming computation: recreating mesh");
        discretization_.refinement() = 0; /* do not refine */
        discretization_.prepare();
        discretization_.triangulation().load(base_name_ + "-checkpoint.mesh");

        print_info("preparing compute kernels");
        prepare_compute_kernels();

        print_info("resuming computation: loading state vector");
        U.reinit(offline_data_.vector_partitioner());
        do_resume(
            offline_data_, base_name_, U, t, output_cycle, mpi_communicator_);
        t_initial_ = t;

        /* Workaround: Reinitialize Quantities with correct output cycle: */
        quantities_.prepare(base_name_, output_cycle);

        /* Remove outdated refinement timestamps: */
        const auto new_end =
            std::remove_if(t_refinements_.begin(),
                           t_refinements_.end(),
                           [&](const Number &t_ref) { return (t >= t_ref); });
        t_refinements_.erase(new_end, t_refinements_.end());

      } else {

        print_info("creating mesh");
        discretization_.prepare();

        print_info("preparing compute kernels");
        prepare_compute_kernels();

        print_info("interpolating initial values");
        U.reinit(offline_data_.vector_partitioner());
        U = initial_values_.interpolate();
#ifdef DEBUG
        /* Poison constrained degrees of freedom: */
        const unsigned int n_relevant = offline_data_.n_locally_relevant();
        const auto &partitioner = offline_data_.scalar_partitioner();
        for (unsigned int i = 0; i < n_relevant; ++i) {
          if (offline_data_.affine_constraints().is_constrained(
                  partitioner->local_to_global(i)))
            U.write_tensor(dealii::Tensor<1, dim + 2, Number>() *
                               std::numeric_limits<Number>::signaling_NaN(),
                           i);
        }
#endif
      }
    }

    print_info("entering main loop");
    computing_timer_["time loop"].start();

    /* Loop: */

    unsigned int cycle = 1;
    for (;; ++cycle) {

#ifdef DEBUG_OUTPUT
      std::cout << "\n\n###   cycle = " << cycle << "   ###\n\n" << std::endl;
#endif

      /* Accumulate quantities of interest: */

      if (enable_compute_quantities_) {
        Scope scope(computing_timer_,
                    "time step [P] X - accumulate quantities");
        quantities_.accumulate(U, t);
      }

      /* Perform output: */

      if (t >= output_cycle * output_granularity_) {
        if (write_output_files) {
          output(U, base_name_ + "-solution", t, output_cycle);
          if (enable_compute_error_) {
            const auto analytic = initial_values_.interpolate(t);
            output(
                analytic, base_name_ + "-analytic_solution", t, output_cycle);
          }
        }
        if (enable_compute_quantities_ &&
            (output_cycle % output_quantities_multiplier_ == 0) &&
            (output_cycle > 0)) {
          Scope scope(computing_timer_,
                      "time step [P] X - write out quantities");
          quantities_.write_out(U, t, output_cycle);
        }
        ++output_cycle;
      }

      /* Perform global refinement: */

      const auto new_end = std::remove_if(
          t_refinements_.begin(),
          t_refinements_.end(),
          [&](const Number &t_ref) {
            if (t < t_ref)
              return false;

            computing_timer_["time loop"].stop();
            Scope scope(computing_timer_, "(re)initialize data structures");

            print_info("performing global refinement");

            SolutionTransfer<dim, Number> solution_transfer(offline_data_,
                                                            hyperbolic_system_);

            auto &triangulation = discretization_.triangulation();
            for (auto &cell : triangulation.active_cell_iterators())
              cell->set_refine_flag();
            triangulation.prepare_coarsening_and_refinement();

            solution_transfer.prepare_for_interpolation(U);

            triangulation.execute_coarsening_and_refinement();
            prepare_compute_kernels();

            solution_transfer.interpolate(U);

            computing_timer_["time loop"].start();
            return true;
          });
      t_refinements_.erase(new_end, t_refinements_.end());

      /* Break if we have reached the final time: */

      if (t >= t_final_)
        break;

      /* Do a time step: */

      const auto tau = time_integrator_.step(U, t, cycle);
      t += tau;

      /* Print and record cycle statistics: */

      if (t >= output_cycle * output_granularity_)
        print_cycle_statistics(cycle, t, output_cycle, /*logfile*/ true);

      if (terminal_update_interval_ != 0 &&
          cycle % terminal_update_interval_ == 0)
        print_cycle_statistics(cycle, t, output_cycle);
    } /* end of loop */

    /* We have actually performed one cycle less. */
    --cycle;

    computing_timer_["time loop"].stop();

    /* Write final timing statistics to screen and logfile: */
    if (terminal_update_interval_ != 0) {
      print_cycle_statistics(
          cycle, t, output_cycle, /*logfile*/ false, /*final_time=*/true);
    }
    print_cycle_statistics(
        cycle, t, output_cycle, /*logfile*/ true, /*final_time=*/true);

    if (enable_compute_error_) {
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

    constexpr auto problem_dimension = HyperbolicSystem::problem_dimension<dim>;

    /* Compute L_inf norm: */

    Vector<float> difference_per_cell(
        discretization_.triangulation().n_active_cells());

    Number linf_norm = 0.;
    Number l1_norm = 0;
    Number l2_norm = 0;

    const auto analytic = initial_values_.interpolate(t);

    scalar_type analytic_component;
    scalar_type error_component;
    analytic_component.reinit(offline_data_.scalar_partitioner());
    error_component.reinit(offline_data_.scalar_partitioner());

    for (unsigned int i = 0; i < problem_dimension; ++i) {

      /* Compute norms of analytic solution: */

      analytic.extract_component(analytic_component, i);

      const Number linf_norm_analytic = Utilities::MPI::max(
          analytic_component.linfty_norm(), mpi_communicator_);

      VectorTools::integrate_difference(offline_data_.dof_handler(),
                                        analytic_component,
                                        Functions::ZeroFunction<dim, Number>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L1_norm);

      const Number l1_norm_analytic =
          Utilities::MPI::sum(difference_per_cell.l1_norm(), mpi_communicator_);

      VectorTools::integrate_difference(offline_data_.dof_handler(),
                                        analytic_component,
                                        Functions::ZeroFunction<dim, Number>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L2_norm);

      const Number l2_norm_analytic = Number(std::sqrt(Utilities::MPI::sum(
          std::pow(difference_per_cell.l2_norm(), 2), mpi_communicator_)));

      /* Compute norms of error: */

      U.extract_component(error_component, i);
      /* Populate constrained dofs due to periodicity: */
      offline_data_.affine_constraints().distribute(error_component);
      error_component.update_ghost_values();
      error_component -= analytic_component;

      const Number linf_norm_error =
          Utilities::MPI::max(error_component.linfty_norm(), mpi_communicator_);

      VectorTools::integrate_difference(offline_data_.dof_handler(),
                                        error_component,
                                        Functions::ZeroFunction<dim, Number>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L1_norm);

      const Number l1_norm_error =
          Utilities::MPI::sum(difference_per_cell.l1_norm(), mpi_communicator_);

      VectorTools::integrate_difference(offline_data_.dof_handler(),
                                        error_component,
                                        Functions::ZeroFunction<dim, Number>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L2_norm);

      const Number l2_norm_error = Number(std::sqrt(Utilities::MPI::sum(
          std::pow(difference_per_cell.l2_norm(), 2), mpi_communicator_)));

      if (linf_norm_analytic >= 1.0e-10)
        linf_norm += linf_norm_error / linf_norm_analytic;
      if (l1_norm_analytic >= 1.0e-10)
        l1_norm += l1_norm_error / l1_norm_analytic;
      if (l2_norm_analytic >= 1.0e-10)
        l2_norm += l2_norm_error / l2_norm_analytic;
    }

    if (mpi_rank_ != 0)
      return;

    logfile_ << std::endl << "Computed errors:" << std::endl << std::endl;

    logfile_ << "Normalized consolidated Linf, L1, and L2 errors at "
             << "final time" << std::endl;
    logfile_ << "#dofs = " << offline_data_.dof_handler().n_dofs() << std::endl;
    logfile_ << "t     = " << t << std::endl;
    logfile_ << "Linf  = " << linf_norm << std::endl;
    logfile_ << "L1    = " << l1_norm << std::endl;
    logfile_ << "L2    = " << l2_norm << std::endl;

    std::cout << "Normalized consolidated Linf, L1, and L2 errors at "
              << "final time" << std::endl;
    std::cout << "#dofs = " << offline_data_.dof_handler().n_dofs()
              << std::endl;
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
        (cycle % output_full_multiplier_ == 0) && enable_output_full_;
    const bool do_levelsets =
        (cycle % output_levelsets_multiplier_ == 0) && enable_output_levelsets_;
    const bool do_checkpointing =
        (cycle % output_checkpoint_multiplier_ == 0) && enable_checkpointing_;

    /* There is nothing to do: */
    if (!(do_full_output || do_levelsets || do_checkpointing))
      return;

    /* Data output: */
    if (do_full_output || do_levelsets) {
      Scope scope(computing_timer_, "time step [P] Y - output vtu");
      print_info("scheduling output");

      postprocessor_.compute(U);
      vtu_output_.schedule_output(
          U, name, t, cycle, do_full_output, do_levelsets);
    }

    /* Checkpointing: */
    if (do_checkpointing) {
      Scope scope(computing_timer_, "time step [P] Z - checkpointing");
      print_info("scheduling checkpointing");

      do_checkpoint(offline_data_, base_name_, U, t, cycle, mpi_communicator_);
    }
  }


  /*
   * Output and logging related functions:
   */


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_parameters(std::ostream &stream)
  {
    if (mpi_rank_ != 0)
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

    stream << "SIMD width == " << VectorizedArray<Number>::size() << std::endl;

#ifdef USE_CUSTOM_POW
    stream << "serial pow == broadcasted pow(Vec4f)/pow(Vec2d)" << std::endl;
#else
    stream << "serial pow == std::pow"<< std::endl;
#endif

    /* clang-format on */

    stream << std::endl;
    stream << std::endl << "Run time parameters:" << std::endl << std::endl;
    ParameterAcceptor::prm.print_parameters(
        stream, ParameterHandler::OutputStyle::ShortText);
    stream << std::endl;

    /* Also print out parameters to a prm file: */

    std::ofstream output(base_name_ + "-parameters.prm");
    ParameterAcceptor::prm.print_parameters(output, ParameterHandler::Text);
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_mpi_partition(std::ostream &stream)
  {
    /*
     * Fixme: this conversion to double is really not elegant. We should
     * improve the Utilities::MPI::min_max_avg function in deal.II to
     * handle different data types
     */

    std::vector<double> values = {
        (double)offline_data_.n_export_indices(),
        (double)offline_data_.n_locally_internal(),
        (double)offline_data_.n_locally_owned(),
        (double)offline_data_.n_locally_relevant(),
        (double)offline_data_.n_export_indices() /
            (double)offline_data_.n_locally_relevant(),
        (double)offline_data_.n_locally_internal() /
            (double)offline_data_.n_locally_relevant(),
        (double)offline_data_.n_locally_owned() /
            (double)offline_data_.n_locally_relevant()};

    const auto data = Utilities::MPI::min_max_avg(values, mpi_communicator_);

    if (mpi_rank_ != 0)
      return;

    std::ostringstream output;

    unsigned int n = dealii::Utilities::needed_digits(n_mpi_processes_);

    const auto print_snippet = [&output, n](const std::string &name,
                                            const auto &values) {
      output << name << ": ";
      output << std::setw(9) << (unsigned int)values.min          //
             << " [p" << std::setw(n) << values.min_index << "] " //
             << std::setw(9) << (unsigned int)values.avg << " "   //
             << std::setw(9) << (unsigned int)values.max          //
             << " [p" << std::setw(n) << values.max_index << "]"; //
    };

    const auto print_percentages = [&output, n](const auto &percentages) {
      output << std::endl << "                  ";
      output << "  (" << std::setw(3) << std::setprecision(2)
             << percentages.min * 100 << "% )"
             << " [p" << std::setw(n) << percentages.min_index << "] "
             << "   (" << std::setw(3) << std::setprecision(2)
             << percentages.avg * 100 << "% )"
             << " "
             << "   (" << std::setw(3) << std::setprecision(2)
             << percentages.max * 100 << "% )"
             << " [p" << std::setw(n) << percentages.max_index << "]";
    };

    output << std::endl << std::endl << "Partition:   ";
    print_snippet("exp", data[0]);
    print_percentages(data[4]);

    output << std::endl << "             ";
    print_snippet("int", data[1]);
    print_percentages(data[5]);

    output << std::endl << "             ";
    print_snippet("own", data[2]);
    print_percentages(data[6]);

    output << std::endl << "             ";
    print_snippet("rel", data[3]);

    stream << output.str() << std::endl;
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_memory_statistics(std::ostream &stream)
  {
    Utilities::System::MemoryStats stats;
    Utilities::System::get_memory_stats(stats);

    Utilities::MPI::MinMaxAvg data =
        Utilities::MPI::min_max_avg(stats.VmRSS / 1024., mpi_communicator_);

    if (mpi_rank_ != 0)
      return;

    std::ostringstream output;

    unsigned int n = dealii::Utilities::needed_digits(n_mpi_processes_);

    output << "\nMemory:      [MiB]"                          //
           << std::setw(8) << data.min                        //
           << " [p" << std::setw(n) << data.min_index << "] " //
           << std::setw(8) << data.avg << " "                 //
           << std::setw(8) << data.max                        //
           << " [p" << std::setw(n) << data.max_index << "]"; //

    stream << output.str() << std::endl;
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_timers(std::ostream &stream)
  {
    std::vector<std::ostringstream> output(computing_timer_.size());

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
          Utilities::MPI::min_max_avg(timer.wall_time(), mpi_communicator_);

      constexpr auto eps = std::numeric_limits<double>::epsilon();
      /*
       * Cut off at 99.9% to avoid silly percentages cluttering up the
       * output.
       */
      const auto skew_negative = std::max(
          100. * (wall_time.min - wall_time.avg) / wall_time.avg - eps, -99.9);
      const auto skew_positive = std::min(
          100. * (wall_time.max - wall_time.avg) / wall_time.avg + eps, 99.9);

      stream << std::setprecision(2) << std::fixed << std::setw(8)
             << wall_time.avg << "s [sk: " << std::setprecision(1)
             << std::setw(5) << std::fixed << skew_negative << "%/"
             << std::setw(4) << std::fixed << skew_positive << "%]";
      unsigned int n = dealii::Utilities::needed_digits(n_mpi_processes_);
      stream << " [p" << std::setw(n) << wall_time.min_index << "/"
             << wall_time.max_index << "]";
    };

    const auto cpu_time_statistics = Utilities::MPI::min_max_avg(
        computing_timer_["time loop"].cpu_time(), mpi_communicator_);
    const double total_cpu_time = cpu_time_statistics.sum;

    const auto print_cpu_time =
        [&](auto &timer, auto &stream, bool percentage) {
          const auto cpu_time =
              Utilities::MPI::min_max_avg(timer.cpu_time(), mpi_communicator_);

          stream << std::setprecision(2) << std::fixed << std::setw(9)
                 << cpu_time.sum << "s ";

          if (percentage)
            stream << "(" << std::setprecision(1) << std::setw(4)
                   << 100. * cpu_time.sum / total_cpu_time << "%)";
        };

    auto jt = output.begin();
    for (auto &it : computing_timer_)
      *jt++ << "  " << it.first;
    equalize();

    jt = output.begin();
    for (auto &it : computing_timer_)
      print_wall_time(it.second, *jt++);
    equalize();

    jt = output.begin();
    bool compute_percentages = false;
    for (auto &it : computing_timer_) {
      print_cpu_time(it.second, *jt++, compute_percentages);
      if (it.first.find("time loop") == 0)
        compute_percentages = true;
    }
    equalize();

    if (mpi_rank_ != 0)
      return;

    stream << std::endl << "Timer statistics:\n";
    for (auto &it : output)
      stream << it.str() << std::endl;
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_throughput(unsigned int cycle,
                                               Number t,
                                               std::ostream &stream,
                                               bool update,
                                               bool final_time)
  {
    /*
     * @fixme The global state kept in this function should be refactored
     * into its own class object.
     */
    static struct Data {
      unsigned int cycle = 0;
      double t = 0.;
      double cpu_time_sum = 0.;
      double cpu_time_avg = 0.;
      double cpu_time_min = 0.;
      double cpu_time_max = 0.;
      double wall_time = 0.;
    } previous, current;

    static double time_per_second_exp = 0.;

    /* Update statistics: */

    {
      if (update)
        previous = current;

      current.cycle = cycle;
      current.t = t;

      const auto wall_time_statistics = Utilities::MPI::min_max_avg(
          computing_timer_["time loop"].wall_time(), mpi_communicator_);
      current.wall_time = wall_time_statistics.max;

      const auto cpu_time_statistics = Utilities::MPI::min_max_avg(
          computing_timer_["time loop"].cpu_time(), mpi_communicator_);
      current.cpu_time_sum = cpu_time_statistics.sum;
      current.cpu_time_avg = cpu_time_statistics.avg;
      current.cpu_time_min = cpu_time_statistics.min;
      current.cpu_time_max = cpu_time_statistics.max;
    }

    if (final_time)
      previous = Data();

    /* Take averages: */

    double delta_cycles = current.cycle - previous.cycle;
    const double cycles_per_second =
        delta_cycles / (current.wall_time - previous.wall_time);

    const double wall_m_dofs_per_sec =
        delta_cycles * ((double)offline_data_.dof_handler().n_dofs()) / 1.e6 /
        (current.wall_time - previous.wall_time);

    const double cpu_m_dofs_per_sec =
        delta_cycles * ((double)offline_data_.dof_handler().n_dofs()) / 1.e6 /
        (current.cpu_time_sum - previous.cpu_time_sum);

    double cpu_time_skew = (current.cpu_time_max - current.cpu_time_min - //
                            previous.cpu_time_max + previous.cpu_time_min) /
                           delta_cycles;
    /* avoid printing small negative numbers: */
    cpu_time_skew = std::max(0., cpu_time_skew);

    const double cpu_time_skew_percentage =
        cpu_time_skew * delta_cycles /
        (current.cpu_time_avg - previous.cpu_time_avg);

    const double delta_time =
        (current.t - previous.t) / (current.cycle - previous.cycle);
    const double time_per_second =
        (current.t - previous.t) / (current.wall_time - previous.wall_time);

    /* Print Jean-Luc and Martin metrics: */

    std::ostringstream output;

    /* clang-format off */
    output << std::endl;

    output << "Throughput:\n  CPU : "
           << std::setprecision(4) << std::fixed << cpu_m_dofs_per_sec
           << " MQ/s  ("
           << std::scientific << 1. / cpu_m_dofs_per_sec * 1.e-6
           << " s/Qdof/cycle)" << std::endl;

    output << "        [cpu time skew: "
           << std::setprecision(2) << std::scientific << cpu_time_skew
           << "s/cycle ("
           << std::setprecision(1) << std::setw(4) << std::setfill(' ') << std::fixed
           << 100. * cpu_time_skew_percentage
           << "%)]" << std::endl;

    output << "  WALL: "
           << std::setprecision(4) << std::fixed << wall_m_dofs_per_sec
           << " MQ/s  ("
           << std::scientific << 1. / wall_m_dofs_per_sec * 1.e-6
           << " s/Qdof/cycle)  ("
           << std::setprecision(2) << std::fixed << cycles_per_second
           << " cycles/s)" << std::endl;

    output << "        [ CFL = "
           << std::setprecision(2) << std::fixed << hyperbolic_module_.cfl()
           << " ("
           << std::setprecision(0) << std::fixed << hyperbolic_module_.n_restarts()
           << " rsts) ("
           << std::setprecision(0) << std::fixed << hyperbolic_module_.n_warnings()
           << " warn) ]" << std::endl;

    output << "        [ dt = "
           << std::scientific << std::setprecision(2) << delta_time
           << " ( "
           << time_per_second
           << " dt/s) ]" << std::endl;
    /* clang-format on */

    /* and print an ETA */
    time_per_second_exp = 0.8 * time_per_second_exp + 0.2 * time_per_second;
    unsigned int eta =
        static_cast<unsigned int>((t_final_ - t) / time_per_second_exp);

    output << "\n  ETA : ";

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
    output << minutes << " min";

    if (mpi_rank_ != 0)
      return;

    stream << output.str() << std::endl;
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_info(const std::string &header)
  {
    if (mpi_rank_ != 0)
      return;

    std::cout << "[INFO] " << header << std::endl;
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_head(const std::string &header,
                                         const std::string &secondary,
                                         std::ostream &stream)
  {
    if (mpi_rank_ != 0)
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

    std::ostringstream primary;
    if (final_time) {
      primary << "FINAL  (cycle " << Utilities::int_to_string(cycle, 6) << ")";
    } else {
      primary << "Cycle  " << Utilities::int_to_string(cycle, 6) //
              << "  (" << std::fixed << std::setprecision(1)     //
              << t / t_final_ * 100 << "%)";
    }

    std::ostringstream secondary;
    secondary << "at time t = " << std::setprecision(8) << std::fixed << t;

    print_head(primary.str(), secondary.str(), output);

    output << "Information: (HYP) " << hyperbolic_system_.problem_name        //
           << "\n             [" << base_name_ << "] with "                   //
           << offline_data_.dof_handler().n_dofs() << " Qdofs on "            //
           << n_mpi_processes_ << " ranks / " << MultithreadInfo::n_threads() //
           << " threads."                                                     //
           << "\n             Last output cycle " << output_cycle - 1         //
           << " at t = " << output_granularity_ * (output_cycle - 1) << "\n"; //

    print_memory_statistics(output);
    print_timers(output);
    print_throughput(cycle,
                     t,
                     output,
                     /*update*/ !write_to_logfile,
                     /*final_time*/ final_time);

    if (mpi_rank_ == 0) {
      if (write_to_logfile) {
        logfile_ << "\n" << output.str() << std::flush;
      } else {
#ifndef DEBUG_OUTPUT
        std::cout << "\033[2J\033[H";
#endif
        std::cout << output.str() << std::flush;
      }
    }
  }

} // namespace ryujin
