//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/computing_timer.h>
#include <ryujin/base/gpu.h>
#include <ryujin/base/version_info.h>
#include <ryujin/driver/logger.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/utilities.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>

#ifdef WITH_OPENMP
#include "omp.h"
#endif

using namespace dealii;

namespace ryujin
{
  template <int dim, typename Number>
  Logger<dim, Number>::Logger(
      const Interface::Simulation<dim, Number> &simulation,
      const std::string &subsection /*= "/Logger"*/)
      : ParameterAcceptor(subsection)
      , Interface::Task<Number>("logger", TaskPhase::post_step)
      , simulation_(simulation)
      , mpi_ensemble_(simulation.mpi_ensemble())
      , n_global_dofs_(0)
      , n_devices_(0)
      , last_terminal_output_(0.)
      , time_per_second_exp_(0.)
  {
    terminal_update_interval_ = 5;
    add_parameter("terminal update interval",
                  terminal_update_interval_,
                  "Number of seconds after which output statistics are "
                  "recomputed and printed on the terminal. Setting the "
                  "interval to zero disables terminal output.");

    terminal_correct_for_hyperthreading_ = true;
    add_parameter(
        "terminal correct for hyperthreading",
        terminal_correct_for_hyperthreading_,
        "If set to true, the CPU throughput is corrected by dividing the total "
        "consumed CPU time by a factor of 2. This correction is only active if "
        "the number of threads (per MPI rank) is 2.");

    debug_command_ = "";
    add_parameter("debug command",
                  debug_command_,
                  "If set to a nonempty string then the host environment's "
                  "command processor is invoked via std::system() with the "
                  "specified string as command parameter.");

    debug_filename_ = "";
    add_parameter("debug filename",
                  debug_filename_,
                  "If set to a nonempty string then we output the contents of "
                  "this file at the end. This is mainly useful in the "
                  "testsuite to output files we wish to compare");
  }


  /*
   * ---------------------------------------------------------------------------
   * Task:
   * ---------------------------------------------------------------------------
   */


  template <int dim, typename Number>
  void Logger<dim, Number>::initialize(TaskContext<Number> &context)
  {
    last_terminal_output_ = terminal_update_interval_ == Number(0.)
                                ? std::numeric_limits<Number>::max()
                                : Number(0.);

    /* Attach log file and record runtime parameters: */

    if (mpi_ensemble_.world_rank() == 0)
      logfile_.open(context.base_name + ".log");

    print_parameters(context.base_name, logfile_);
  }


  template <int dim, typename Number>
  void Logger<dim, Number>::start(TaskContext<Number> & /*context*/)
  {
    print_device_information();
  }


  template <int dim, typename Number>
  void Logger<dim, Number>::prepare(TaskContext<Number> & /*context*/)
  {
    print_mpi_partition(logfile_);

    if (mpi_ensemble_.ensemble_rank() == 0)
      n_global_dofs_ = dealii::Utilities::MPI::sum(
          simulation_.offline_data().dof_handler().n_dofs(),
          mpi_ensemble_.ensemble_leader_communicator());
  }


  template <int dim, typename Number>
  void Logger<dim, Number>::execute(TaskContext<Number> &context)
  {
    const bool write_to_logfile =
        (terminal_update_interval_ != Number(0.)) && /* suppress output */
        context.timer_tick;

    const bool update_terminal =
        (context.wall_time >=
         last_terminal_output_ + terminal_update_interval_);

    if (!(write_to_logfile || update_terminal))
      return;

    ComputingTimer::Scope scope("time step [X] _ - synchronization barriers");
    print_cycle_statistics(context, /*logfile*/ write_to_logfile);
    last_terminal_output_ = context.wall_time;
  }


  template <int dim, typename Number>
  void Logger<dim, Number>::finalize(TaskContext<Number> &context)
  {
    /* Write final timing statistics to screen and logfile: */

    if (terminal_update_interval_ != Number(0.))
      print_cycle_statistics(context, /*logfile*/ true, /*final*/ true);

    print_debug_output();
  }


  template <int dim, typename Number>
  void Logger<dim, Number>::print_debug_output()
  {
    if (mpi_ensemble_.world_rank() != 0)
      return;

    if (debug_command_ != "") {
      auto result [[maybe_unused]] = std::system(debug_command_.c_str());
    }

    if (debug_filename_ != "") {
      std::ifstream f(debug_filename_);
      if (f.is_open())
        std::cout << f.rdbuf();
    }
  }


  /*
   * ---------------------------------------------------------------------------
   * Internal methods:
   * ---------------------------------------------------------------------------
   */


  template <int dim, typename Number>
  void Logger<dim, Number>::print_parameters(const std::string &base_name,
                                             std::ostream &stream)
  {
    if (mpi_ensemble_.world_rank() != 0)
      return;

    /* Output commit and library information: */

    print_revision_and_version(stream);

    /* Print run time parameters: */

    stream << std::endl << "Run time parameters:" << std::endl << std::endl;
    ParameterAcceptor::prm.print_parameters(
        stream, ParameterHandler::OutputStyle::ShortPRM);
    stream << std::endl;

    /* Also print out parameters to a prm file: */

    std::ofstream output(base_name + "-parameters.prm");
    ParameterAcceptor::prm.print_parameters(output, ParameterHandler::ShortPRM);
  }


  template <int dim, typename Number>
  void Logger<dim, Number>::print_mpi_partition(std::ostream &stream)
  {
    /*
     * Fixme: this conversion to double is really not elegant. We should
     * improve the Utilities::MPI::min_max_avg function in deal.II to
     * handle different data types
     */

    // NOLINTBEGIN
    std::vector<double> values = {
        (double)simulation_.offline_data().n_export_indices(),
        (double)simulation_.offline_data().n_locally_internal(),
        (double)simulation_.offline_data().n_locally_owned(),
        (double)simulation_.offline_data().n_locally_relevant(),
        (double)simulation_.offline_data().n_export_indices() /
            (double)simulation_.offline_data().n_locally_relevant(),
        (double)simulation_.offline_data().n_locally_internal() /
            (double)simulation_.offline_data().n_locally_relevant(),
        (double)simulation_.offline_data().n_locally_owned() /
            (double)simulation_.offline_data().n_locally_relevant()};
    // NOLINTEND

    const auto data =
        Utilities::MPI::min_max_avg(values, mpi_ensemble_.world_communicator());

    if (mpi_ensemble_.world_rank() != 0)
      return;

    std::ostringstream output;

    unsigned int n =
        dealii::Utilities::needed_digits(mpi_ensemble_.n_world_ranks());

    const auto print_snippet = [&output, n](const std::string &name,
                                            const auto &values) {
      output << name << ": ";
      // NOLINTBEGIN
      output << std::setw(9) << (unsigned int)values.min          //
             << " [p" << std::setw(n) << values.min_index << "] " //
             << std::setw(9) << (unsigned int)values.avg << " "   //
             << std::setw(9) << (unsigned int)values.max          //
             << " [p" << std::setw(n) << values.max_index << "]"; //
      // NOLINTEND
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
  void Logger<dim, Number>::print_device_information()
  {
    /* Device information is only meaningful for a device build: */
    if constexpr (!have_separate_memory_spaces)
      return;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    /*
     * Determine how many distinct devices we are running on: Multiple
     * ranks might be bound to the same device, so counting ranks would
     * overreport. We thus count distinct (hostname, device id) pairs.
     */
    {
      using Device = std::pair<std::string, int>;

      const Device local{dealii::Utilities::System::get_hostname(),
                         Kokkos::device_id()};
      const auto all = dealii::Utilities::MPI::all_gather(
          mpi_ensemble_.world_communicator(), local);
      n_devices_ = std::set<Device>(all.begin(), all.end()).size();
    }
#endif

    if (mpi_ensemble_.world_rank() != 0)
      return;

    std::ostringstream output;

    const auto entry [[maybe_unused]] =
        [&output](const std::string &label) -> std::ostream & {
      output << std::endl
             << "             " << std::left << std::setw(22) << label;
      return output;
    };

    output << std::endl
           << std::endl
           << "Device:      " << std::left << std::setw(22) << "backend"
           << Kokkos::DefaultExecutionSpace::name();

    /*
     * Kokkos does not provide a portable device property class. We thus
     * have to query the individual backends by hand:
     */

#if defined(KOKKOS_ENABLE_CUDA)
    const auto &prop = Kokkos::Cuda{}.cuda_device_prop();

    entry("device") << prop.name;
    entry("compute capability") << prop.major << "." << prop.minor;
    entry("device id") << Kokkos::device_id() << " of "
                       << Kokkos::num_devices();
    entry("devices in use")
        << n_devices_ << " on " << mpi_ensemble_.n_world_ranks() << " ranks";
    entry("multiprocessors") << prop.multiProcessorCount;
    entry("warp size") << prop.warpSize << " (hardware), " << warp_size
                       << " (ryujin::warp_size)";
    entry("threads per SM")
        << prop.maxThreadsPerMultiProcessor << " ("
        << prop.maxThreadsPerMultiProcessor / prop.warpSize << " warps)";
    entry("threads per block") << prop.maxThreadsPerBlock;
    entry("concurrency") << Kokkos::DefaultExecutionSpace{}.concurrency();
    entry("registers per SM") << prop.regsPerBlock;
    entry("shared memory per SM")
        << prop.sharedMemPerMultiprocessor / 1024 << " KiB";
    entry("L2 cache") << prop.l2CacheSize / 1024 / 1024 << " MiB";
    entry("global memory") << prop.totalGlobalMem / 1024 / 1024 << " MiB";

#elif defined(KOKKOS_ENABLE_HIP)
    const auto &prop = Kokkos::HIP::hip_device_prop();

    entry("device") << prop.name;
    entry("architecture") << prop.gcnArchName;
    entry("device id") << Kokkos::device_id() << " of "
                       << Kokkos::num_devices();
    entry("devices in use")
        << n_devices_ << " on " << mpi_ensemble_.n_world_ranks() << " ranks";
    entry("compute units") << prop.multiProcessorCount;
    entry("warp size") << prop.warpSize << " (hardware), " << warp_size
                       << " (ryujin::warp_size)";
    entry("threads per CU")
        << prop.maxThreadsPerMultiProcessor << " ("
        << prop.maxThreadsPerMultiProcessor / prop.warpSize << " warps)";
    entry("threads per block") << prop.maxThreadsPerBlock;
    entry("concurrency") << Kokkos::DefaultExecutionSpace{}.concurrency();
    entry("registers per block") << prop.regsPerBlock;
    entry("shared memory per block") << prop.sharedMemPerBlock / 1024 << " KiB";
    entry("L2 cache") << prop.l2CacheSize / 1024 / 1024 << " MiB";
    entry("global memory") << prop.totalGlobalMem / 1024 / 1024 << " MiB";
#endif

    logfile_ << output.str() << std::endl;
  }


  template <int dim, typename Number>
  void Logger<dim, Number>::print_head(const std::string &header,
                                       const std::string &secondary,
                                       std::ostream &stream)
  {
    if (mpi_ensemble_.world_rank() != 0)
      return;

    const int header_size = header.size();
    const auto padded_header =
        std::string(std::max(0, 34 - header_size) / 2, ' ') + header +
        std::string(std::max(0, 35 - header_size) / 2, ' ');

    const int secondary_size = secondary.size();
    const auto padded_secondary =
        std::string(std::max(0, 34 - secondary_size) / 2, ' ') + secondary +
        std::string(std::max(0, 35 - secondary_size) / 2, ' ');

    /* clang-format off */
    stream << "\n";
    stream << "    ####################################################\n";
    stream << "    #########"     <<  padded_header   <<     "#########\n";
    stream << "    #########"     << padded_secondary <<     "#########\n";
    stream << "    ####################################################\n";
    stream << std::endl;
    /* clang-format on */
  }


  template <int dim, typename Number>
  void
  Logger<dim, Number>::print_information(const TaskContext<Number> &context,
                                         std::ostream &stream)
  {
    static const std::string backend_name = [] {
      const auto precision = [] {
        if constexpr (std::is_same_v<Number, double>)
          return std::string("FP64");
        else if constexpr (std::is_same_v<Number, float>)
          return std::string("FP32");
        else
          __builtin_trap();
      }();

      constexpr auto simd_size = VectorizedArray<Number>::size();

      if constexpr (have_separate_memory_spaces)
        return "GPU, " + precision + ", warp size " + std::to_string(warp_size);
      else if constexpr (simd_size == 1)
        return "scalar, " + precision;
      else
        return "SIMD, " + precision + ", width " + std::to_string(simd_size);
    }();

    stream << "Information: (HYP) "
           << simulation_.hyperbolic_system().problem_name();
    if (!simulation_.parabolic_system().is_identity()) {
      stream << "\n             (PAR) "
             << simulation_.parabolic_system().problem_name();
    }
    stream << "\n             [" << context.base_name << "] ";
    if (mpi_ensemble_.n_ensembles() > 1) {
      stream << mpi_ensemble_.n_ensembles() << " ensembles ";
    }
    stream << "with " << n_global_dofs_ << " Qdofs on "
           << mpi_ensemble_.n_world_ranks() << " ranks "
#if defined(WITH_OPENMP)
           << "/ " << omp_get_max_threads() << " threads "
#ifndef WITH_DEAL_II_THREADS
           << "[serial dealii] "
#endif
#elif defined(WITH_DEAL_II_THREADS)
           << "/ " << MultithreadInfo::n_threads() << " threads "
#endif
           << "<" << backend_name << ">\n";

    stream << "             Last output cycle " //
           << context.timer_cycle - 1           //
           << " at t = "
           << context.timer_granularity * (context.timer_cycle - 1) //
           << "  [ tasks: ";

    for (const auto *task : context.tasks)
      stream << task->name() << (task == context.tasks.back() ? " " : ", ");

    stream << "]\n";
  }


  template <int dim, typename Number>
  void Logger<dim, Number>::print_memory_statistics(std::ostream &stream)
  {
    Utilities::System::MemoryStats stats;
    Utilities::System::get_memory_stats(stats);

    Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(
        stats.VmRSS / 1024., mpi_ensemble_.world_communicator());

    if (mpi_ensemble_.world_rank() != 0)
      return;

    std::ostringstream output;

    unsigned int n =
        dealii::Utilities::needed_digits(mpi_ensemble_.n_world_ranks());

    output << "\nMemory:      [MiB]"                          //
           << std::setw(8) << data.min                        //
           << " [p" << std::setw(n) << data.min_index << "] " //
           << std::setw(8) << data.avg << " "                 //
           << std::setw(8) << data.max                        //
           << " [p" << std::setw(n) << data.max_index << "]"; //

    stream << output.str() << std::endl;
  }


  template <int dim, typename Number>
  void Logger<dim, Number>::print_timers(std::ostream &stream)
  {
    std::vector<std::ostringstream> output(ComputingTimer::timers().size());

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
      const auto wall_time = Utilities::MPI::min_max_avg(
          timer.wall_time(), mpi_ensemble_.world_communicator());

      constexpr auto eps = std::numeric_limits<double>::epsilon();
      /*
       * Cut off at 99.9% to avoid silly percentages cluttering up the
       * output.
       */
      const auto skew_negative = std::max(
          100. * (wall_time.min - wall_time.avg) / wall_time.avg - eps, -99.9);
      const auto skew_positive = std::min(
          100. * (wall_time.max - wall_time.avg) / wall_time.avg + eps, 99.9);

      stream << std::setprecision(2) << std::fixed << std::setw(9)
             << wall_time.avg << "s [sk: " << std::setprecision(1)
             << std::setw(5) << std::fixed << skew_negative << "%/"
             << std::setw(4) << std::fixed << skew_positive << "%]";
      unsigned int n =
          dealii::Utilities::needed_digits(mpi_ensemble_.n_world_ranks());
      stream << " [p" << std::setw(n) << wall_time.min_index << "/"
             << wall_time.max_index << "]";
    };

    const auto cpu_time_statistics = Utilities::MPI::min_max_avg(
        ComputingTimer::timer("time loop").cpu_time(),
        mpi_ensemble_.world_communicator());
    const double total_cpu_time = cpu_time_statistics.sum;

    const auto print_cpu_time =
        [&](auto &timer, auto &stream, bool percentage) {
          const auto cpu_time = Utilities::MPI::min_max_avg(
              timer.cpu_time(), mpi_ensemble_.world_communicator());

          stream << std::setprecision(2) << std::fixed << std::setw(12)
                 << cpu_time.sum << "s ";

          if (percentage)
            stream << "(" << std::setprecision(1) << std::setw(4)
                   << 100. * cpu_time.sum / total_cpu_time << "%)";
        };

    auto jt = output.begin();
    for (auto &it : ComputingTimer::timers())
      *jt++ << "  " << it.first;
    equalize();

    jt = output.begin();
    for (auto &it : ComputingTimer::timers())
      print_wall_time(it.second, *jt++);
    equalize();

    jt = output.begin();
    bool compute_percentages = false;
    for (auto &it : ComputingTimer::timers()) {
      print_cpu_time(it.second, *jt++, compute_percentages);
      if (it.first.starts_with("time loop"))
        compute_percentages = true;
    }
    equalize();

    if (mpi_ensemble_.world_rank() != 0)
      return;

    stream << std::endl << "Timer statistics:\n";
    for (auto &it : output)
      stream << it.str() << std::endl;
  }


  template <int dim, typename Number>
  void Logger<dim, Number>::print_throughput(const TaskContext<Number> &context,
                                             std::ostream &stream,
                                             bool final_time)
  {
    const auto cycle = context.cycle;
    const auto t = context.t;

    /* Update statistics: */

    {
      previous_ = current_;

      current_.cycle = cycle;
      current_.t = t;

      const auto wall_time_statistics = Utilities::MPI::min_max_avg(
          ComputingTimer::timer("time loop").wall_time(),
          mpi_ensemble_.world_communicator());
      current_.wall_time = wall_time_statistics.max;

      const auto cpu_time_statistics = Utilities::MPI::min_max_avg(
          ComputingTimer::timer("time loop").cpu_time(),
          mpi_ensemble_.world_communicator());
      current_.cpu_time_sum = cpu_time_statistics.sum;
      current_.cpu_time_avg = cpu_time_statistics.avg;
      current_.cpu_time_min = cpu_time_statistics.min;
      current_.cpu_time_max = cpu_time_statistics.max;

      if constexpr (have_separate_memory_spaces) {
        const auto device_time_statistics = Utilities::MPI::min_max_avg(
            DeviceTimer::seconds(), mpi_ensemble_.world_communicator());
        current_.device_time_sum = device_time_statistics.sum;
      }
    }

    if (final_time)
      previous_ = ThroughputData();

    /* Take averages: */

    double delta_cycles = current_.cycle - previous_.cycle;
    const double cycles_per_second =
        delta_cycles / (current_.wall_time - previous_.wall_time);

    const auto efficiency = simulation_.time_integrator().efficiency();
    const auto n_dofs = static_cast<double>(n_global_dofs_);

    double wall_m_dofs_per_sec = delta_cycles * n_dofs * efficiency / 1.e6 /
                                 (current_.wall_time - previous_.wall_time);

    const double delta_time =
        (current_.t - previous_.t) / (current_.cycle - previous_.cycle);
    const double time_per_second =
        (current_.t - previous_.t) / (current_.wall_time - previous_.wall_time);

    /* Print Jean-Luc and Martin metrics: */

    std::ostringstream output;

    /* clang-format off */
    output << std::endl;

    output << "Throughput:" << std::endl;

    if constexpr (have_separate_memory_spaces) {
      /* Write out some statistics about GPU utilization: */

      const double delta_wall_time = current_.wall_time - previous_.wall_time;
      const double delta_device_time =
          current_.device_time_sum - previous_.device_time_sum;

      const double utilization =
          delta_device_time / (n_devices_ * delta_wall_time);

      const double device_m_dofs_per_sec =
          delta_cycles * n_dofs * efficiency / 1.e6 / delta_device_time;
      output << "  GPU : "
             << std::setprecision(4) << std::fixed << device_m_dofs_per_sec
             << " MQ/s in compute kernels (on "
             << n_devices_ << (n_devices_ == 1 ? " device)" : " devices)")
             << std::endl;

      output << "        ["
             << std::setprecision(2) << std::fixed << delta_device_time
             << "s device / "
             << std::setprecision(2) << std::fixed << delta_wall_time
             << "s wall = "
             << std::setprecision(1) << std::fixed << 100. * utilization
             << "% utilization ]" << std::endl;

    } else {
      /* Write out some statistics about CPU utilization: */

      double cpu_m_dofs_per_sec = delta_cycles * n_dofs * efficiency / 1.e6 /
                                  (current_.cpu_time_sum - previous_.cpu_time_sum);

      /* Determine whether we fudge the CPU timings: */
      const bool fudge_cpu_timings = terminal_correct_for_hyperthreading_ &&
#if defined(WITH_OPENMP)
                                     (omp_get_max_threads() == 2);
#elif defined(WITH_DEAL_II_THREADS)
                                     (MultithreadInfo::n_threads() == 2);
#else
                                     false;
#endif

      if (fudge_cpu_timings)
        cpu_m_dofs_per_sec *= 2.;

      double cpu_time_skew = (current_.cpu_time_max - current_.cpu_time_min - //
                              previous_.cpu_time_max + previous_.cpu_time_min) /
                             delta_cycles;
      /* avoid printing small negative numbers: */
      cpu_time_skew = std::max(0., cpu_time_skew);

      const double cpu_time_skew_percentage =
          cpu_time_skew * delta_cycles /
          (current_.cpu_time_avg - previous_.cpu_time_avg);

      output << "  "
             << (fudge_cpu_timings ? "CPU*: " : "CPU : ")
             << std::setprecision(4) << std::fixed << cpu_m_dofs_per_sec
             << " MQ/s  ("
             << std::scientific << 1. / cpu_m_dofs_per_sec * 1.e-6
             << " s/Qdof/substep)" << std::endl;

      output << "        [cpu time skew: "
             << std::setprecision(2) << std::scientific << cpu_time_skew
             << "s/cycle ("
             << std::setprecision(1) << std::setw(4) << std::setfill(' ') << std::fixed
             << 100. * cpu_time_skew_percentage
             << "%)]" << std::endl;
    }

    output << "  WALL: "
           << std::setprecision(4) << std::fixed << wall_m_dofs_per_sec
           << " MQ/s  ("
           << std::scientific << 1. / wall_m_dofs_per_sec * 1.e-6
           << " s/Qdof/substep)  ("
           << std::setprecision(2) << std::fixed << cycles_per_second
           << " cycles/s)" << std::endl;

    output << "        [ "
           << simulation_.time_integrator().time_stepping_scheme_name()
           << " with CFL = "
           << std::setprecision(2) << std::fixed << simulation_.hyperbolic_module().cfl()
           << " ("
           << std::setprecision(0) << std::fixed << simulation_.hyperbolic_module().n_restarts()
           << "/"
           << std::setprecision(0) << std::fixed << simulation_.parabolic_module().n_restarts()
           << " rsts) ("
           << std::setprecision(0) << std::fixed << simulation_.hyperbolic_module().n_warnings()
           << "/"
           << std::setprecision(0) << std::fixed << simulation_.parabolic_module().n_warnings()
           << " warn) ("
           << std::setprecision(0) << std::fixed << simulation_.hyperbolic_module().n_corrections()
           << "/"
           << std::setprecision(0) << std::fixed << simulation_.parabolic_module().n_corrections()
           << " corr) ]" << std::endl;

    if (!simulation_.parabolic_system().is_identity())
      simulation_.parabolic_module().print_solver_statistics(output);

    output << "        [ dt = "
           << std::scientific << std::setprecision(2) << delta_time
           << " ( "
           << time_per_second
           << " dt/s) ]" << std::endl;
    /* clang-format on */

    /* And print an ETA: */

    time_per_second_exp_ = 0.8 * time_per_second_exp_ + 0.2 * time_per_second;
    auto eta = static_cast<unsigned int>(
        std::max(context.t_final - t, Number(0.)) / time_per_second_exp_);

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

    output << "   (terminal update every " //
           << std::setprecision(2) << std::fixed << terminal_update_interval_
           << "s)";

    if (mpi_ensemble_.world_rank() != 0)
      return;

    stream << output.str() << std::endl;
  }


  template <int dim, typename Number>
  void Logger<dim, Number>::print_cycle_statistics(
      const TaskContext<Number> &context,
      bool write_to_logfile,
      bool final_time)
  {
    const auto cycle = context.cycle;
    const auto t = context.t;

    std::ostringstream output;

    /* Print header: */

    std::ostringstream primary;
    if (final_time) {
      primary << "FINAL  (cycle " << Utilities::int_to_string(cycle, 6) << ")";
    } else {
      primary << "Cycle  " << Utilities::int_to_string(cycle, 6) //
              << "  (" << std::fixed << std::setprecision(1)     //
              << t / context.t_final * 100 << "%)";
    }

    std::ostringstream secondary;
    secondary << "at time t = " << std::setprecision(8) << std::fixed << t;

    print_head(primary.str(), secondary.str(), output);

    /* Print information and statistics: */

    print_information(context, output);
    print_memory_statistics(output);
    print_timers(output);
    print_throughput(context, output, final_time);

    /* Only output on rank 0: */
    if (mpi_ensemble_.world_rank() != 0)
      return;

#ifndef DEBUG_OUTPUT
    std::cout << "\033[2J\033[H";
#endif
    std::cout << output.str() << std::flush;

    if (write_to_logfile) {
      logfile_ << "\n" << output.str() << std::flush;
    }
  }
} // namespace ryujin
