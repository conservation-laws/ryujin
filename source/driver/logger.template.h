//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <driver/dashboard_text.h>
#include <ryujin/base/computing_timer.h>
#include <ryujin/base/gpu.h>
#include <ryujin/base/version_info.h>
#include <ryujin/driver/logger.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/utilities.h>

#include <cmath>
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

#if defined(WITH_FTXUI)
    add_parameter("dashboard",
                  dashboard_settings_.mode,
                  "Draw the terminal dashboard that redraws the terminal in "
                  "place: auto (if stdout reaches the terminal, also through "
                  "mpirun), on, off (print plain text instead).");

    add_parameter("dashboard max width",
                  dashboard_settings_.width,
                  "Maximal width of the terminal dashboard in columns. The "
                  "dashboard fills the terminal up to this width. Setting the "
                  "width to zero removes the limit.");

    add_parameter("dashboard max height",
                  dashboard_settings_.height,
                  "Maximal height of the terminal dashboard in lines. If "
                  "nonzero, the dashboard stretches to fill the terminal up to "
                  "this height. Setting the height to zero keeps the natural "
                  "height of the dashboard.");

    add_parameter(
        "dashboard location",
        dashboard_settings_.location,
        "Location of the terminal dashboard: top left, center, "
        "screen saver (move to a random position every ten seconds, in "
        "the parameter editor only while there is no input).");

    add_parameter("dashboard color scheme",
                  dashboard_settings_.color_scheme,
                  "Color scheme of the terminal dashboard: ryujin, rainbow, "
                  "tamu, monochrome");

    add_parameter("dashboard animation",
                  dashboard_settings_.animation,
                  "Animation in the space around the terminal dashboard: "
                  "sparkle, matthias, none");

    add_parameter("dashboard throughput display",
                  dashboard_settings_.throughput_display,
                  "Display of the wall throughput on the terminal dashboard: "
                  "digital, speedometer");
#endif

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

    /*
     * Record all status messages for the dashboard. If the dashboard is
     * not shown we simply print them as usual:
     */

    print_info_ = context.info;
    context.info = [this, &context](const std::string &message) {
      constexpr std::size_t n_messages = 100;
      auto &messages = dashboard_data_.messages;
      messages.push_back({message, context.cycle, context.t});
      if (messages.size() > n_messages)
        messages.pop_front();

      if (!dashboard_active_ && print_info_)
        print_info_(message);
    };
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

#if defined(WITH_FTXUI)
    /* Play the animation from the start of the main loop on: */
    if (!dashboard_active_ && mpi_ensemble_.world_rank() == 0 &&
        interactive_terminal()) {
      dashboard_.start(dashboard_settings_);
      dashboard_active_ = true;
    }
#endif
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

    if (write_to_logfile || update_terminal) {
      ComputingTimer::Scope scope("time step [X] _ - synchronization barriers");
      print_cycle_statistics(context, /*logfile*/ write_to_logfile);
      last_terminal_output_ = context.wall_time;
    }
  }


  template <int dim, typename Number>
  bool Logger<dim, Number>::interactive_terminal() const
  {
#if defined(DEBUG_OUTPUT)
    return false;
#else
    /* Return whether we are running on an interactive terminal: */
    const auto mode = dashboard_settings_.mode;
    return mode == DashboardMode::on ||
           (mode == DashboardMode::automatic && stdout_reaches_terminal());
#endif
  }


  template <int dim, typename Number>
  void Logger<dim, Number>::finalize(TaskContext<Number> &context)
  {
    /* Write final timing statistics to screen and logfile: */

    if (terminal_update_interval_ != Number(0.))
      print_cycle_statistics(context, /*logfile*/ true, /*final*/ true);

    /* Print status messages after the final statistics as usual: */
    dashboard_active_ = false;

    print_debug_output();
  }


  /*
   * ---------------------------------------------------------------------------
   * Internal methods:
   * ---------------------------------------------------------------------------
   */


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
  void Logger<dim, Number>::print_cycle_statistics(
      const TaskContext<Number> &context,
      bool write_to_logfile,
      bool final_time)
  {
    dashboard_data_.final_time = final_time;
    collect_information(context);
    collect_memory_statistics();
    collect_timers();
    collect_tasks(context);
    collect_throughput(context, final_time);

    /* Only output on rank 0: */
    if (mpi_ensemble_.world_rank() != 0)
      return;

    const auto text = render_text(dashboard_data_);

#if defined(WITH_FTXUI)
    if (interactive_terminal())
      dashboard_.show(dashboard_data_);
    else
#endif
      std::cout << "\033[2J\033[H" << text << std::flush;

    if (write_to_logfile)
      logfile_ << "\n" << text << std::flush;
  }


  template <int dim, typename Number>
  void
  Logger<dim, Number>::collect_information(const TaskContext<Number> &context)
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

    std::string threads;
#if defined(WITH_OPENMP)
    threads = std::to_string(omp_get_max_threads()) + " threads";
#ifndef WITH_DEAL_II_THREADS
    threads += " [serial dealii]";
#endif
#elif defined(WITH_DEAL_II_THREADS)
    threads = std::to_string(MultithreadInfo::n_threads()) + " threads";
#endif

    auto &data = dashboard_data_;
    data.cycle = context.cycle;
    data.t = context.t;
    data.t_final = context.t_final;
    data.hyperbolic_name = simulation_.hyperbolic_system().problem_name();
    data.parabolic_name = simulation_.parabolic_system().is_identity()
                              ? ""
                              : simulation_.parabolic_system().problem_name();
    data.base_name = context.base_name;
    data.backend = backend_name;
    data.threads = threads;
    data.n_global_dofs = n_global_dofs_;
    data.n_ensembles = mpi_ensemble_.n_ensembles();
    data.n_ranks = mpi_ensemble_.n_world_ranks();
  }


  template <int dim, typename Number>
  void Logger<dim, Number>::collect_memory_statistics()
  {
    Utilities::System::MemoryStats stats;
    Utilities::System::get_memory_stats(stats);

    dashboard_data_.memory = Utilities::MPI::min_max_avg(
        stats.VmRSS / 1024., mpi_ensemble_.world_communicator());
  }


  template <int dim, typename Number>
  void Logger<dim, Number>::collect_timers()
  {
    const auto &communicator = mpi_ensemble_.world_communicator();

    const double total_cpu_time =
        Utilities::MPI::min_max_avg(
            ComputingTimer::timer("time loop").cpu_time(), communicator)
            .sum;

    /* Percentages are only meaningful for the timers of the time loop: */
    bool percentage = false;

    auto &timers = dashboard_data_.timers;
    timers.clear();
    for (const auto &[name, timer] : ComputingTimer::timers()) {
      const double cpu_time =
          Utilities::MPI::min_max_avg(timer.cpu_time(), communicator).sum;
      timers.push_back({
          .name = name,
          .wall_time =
              Utilities::MPI::min_max_avg(timer.wall_time(), communicator),
          .cpu_time = cpu_time,
          .fraction = percentage ? cpu_time / total_cpu_time : -1.,
      });
      if (name.starts_with("time loop"))
        percentage = true;
    }
  }


  template <int dim, typename Number>
  void Logger<dim, Number>::collect_tasks(const TaskContext<Number> &context)
  {
    const auto format_duration = [](double seconds) {
      const auto s =
          static_cast<unsigned long>(std::round(std::max(seconds, 0.)));
      std::ostringstream output;
      output << std::setfill('0');
      if (s >= 3600)
        output << s / 3600 << "h " << std::setw(2) << s % 3600 / 60 << "m "
               << std::setw(2) << s % 60 << "s";
      else if (s >= 60)
        output << s / 60 << "m " << std::setw(2) << s % 60 << "s";
      else
        output << s << "s";
      return output.str();
    };

    const auto format_time = [](Number t) {
      std::ostringstream output;
      output << "t = " << std::setprecision(6) << t;
      return output.str();
    };

    const auto phase_name = [](TaskPhase phase) -> std::string {
      switch (phase) {
      case TaskPhase::pre_step:
        return "pre step";
      case TaskPhase::adapt:
        return "adapt";
      case TaskPhase::step:
        return "step";
      case TaskPhase::post_step:
        return "post step";
      }
      __builtin_unreachable();
    };

    auto &tasks = dashboard_data_.tasks;
    tasks.clear();

    for (const auto *task : context.tasks) {
      auto &entry = tasks.emplace_back();
      entry.name = task->name();
      entry.phase = phase_name(task->phase());

      if (!task->has_trigger())
        continue;
      const auto &trigger = task->trigger();

      const auto multiplier = trigger.multiplier();
      switch (trigger.mode()) {
      case TriggerMode::every_cycle:
        entry.trigger = multiplier == 1
                            ? "every cycle"
                            : "every " + std::to_string(multiplier) + " cycles";
        break;
      case TriggerMode::timer_tick:
        entry.trigger =
            multiplier == 1
                ? "every timer tick"
                : "every " + std::to_string(multiplier) + " timer ticks";
        break;
      case TriggerMode::wall_clock:
        entry.trigger =
            "every " + format_duration(trigger.interval()) + " wall time";
        break;
      case TriggerMode::time_points:
        entry.trigger =
            std::to_string(trigger.time_points().size()) + " time points";
        break;
      }

      if (trigger.fired())
        entry.last =
            "cycle " + std::to_string(trigger.last_cycle()) + ", " +
            format_time(trigger.last_t()) + ", " +
            format_duration(context.wall_time - trigger.last_wall_time()) +
            " ago";

      const Number next = trigger.next(context);
      if (next != std::numeric_limits<Number>::max()) {
        switch (trigger.mode()) {
        case TriggerMode::every_cycle:
          entry.next =
              "cycle " + std::to_string(static_cast<unsigned int>(next));
          break;
        case TriggerMode::timer_tick:
        case TriggerMode::time_points:
          entry.next = format_time(next);
          break;
        case TriggerMode::wall_clock:
          entry.next = "in " + format_duration(next - context.wall_time);
          break;
        }
      }

      entry.progress = trigger.progress(context);
    }
  }


  template <int dim, typename Number>
  void
  Logger<dim, Number>::collect_throughput(const TaskContext<Number> &context,
                                          bool final_time)
  {
    /* Update statistics: */

    {
      previous_ = current_;

      current_.cycle = context.cycle;
      current_.t = context.t;

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
        current_.device_time_avg = device_time_statistics.avg;
        current_.device_time_min = device_time_statistics.min;
        current_.device_time_max = device_time_statistics.max;
        dashboard_data_.device_time_min_index =
            device_time_statistics.min_index;
        dashboard_data_.device_time_max_index =
            device_time_statistics.max_index;
      }
    }

    if (final_time)
      previous_ = ThroughputData();

    /* Take averages: */

    const double delta_cycles = current_.cycle - previous_.cycle;
    const double delta_wall_time = current_.wall_time - previous_.wall_time;
    const double delta_q_dofs = delta_cycles *
                                static_cast<double>(n_global_dofs_) *
                                simulation_.time_integrator().efficiency();

    auto &data = dashboard_data_;
    data.device = have_separate_memory_spaces;
    data.n_devices = n_devices_;
    data.wall_m_dofs_per_sec = delta_q_dofs / 1.e6 / delta_wall_time;
    data.cycles_per_second = delta_cycles / delta_wall_time;
    data.delta_wall_time = delta_wall_time;
    data.tau = (current_.t - previous_.t) / delta_cycles;
    data.time_per_second = (current_.t - previous_.t) / delta_wall_time;
    data.terminal_update_interval = terminal_update_interval_;

    if constexpr (have_separate_memory_spaces) {
      /* Statistics about GPU utilization: */

      const double delta_device_time =
          current_.device_time_sum - previous_.device_time_sum;

      data.compute_m_dofs_per_sec = delta_q_dofs / 1.e6 / delta_device_time;
      data.utilization = delta_device_time / (n_devices_ * delta_wall_time);
      data.delta_device_time = delta_device_time;

      /* avoid printing small negative numbers: */
      data.device_time_skew =
          std::max(0.,
                   (current_.device_time_max - current_.device_time_min - //
                    previous_.device_time_max + previous_.device_time_min) /
                       delta_cycles);
      data.device_time_skew_fraction =
          data.device_time_skew * delta_cycles /
          (current_.device_time_avg - previous_.device_time_avg);

    } else {
      /* Statistics about CPU utilization: */

      /* Determine whether we fudge the CPU timings: */
      const bool fudge_cpu_timings = terminal_correct_for_hyperthreading_ &&
#if defined(WITH_OPENMP)
                                     (omp_get_max_threads() == 2);
#elif defined(WITH_DEAL_II_THREADS)
                                     (MultithreadInfo::n_threads() == 2);
#else
                                     false;
#endif

      data.compute_m_dofs_per_sec =
          (fudge_cpu_timings ? 2. : 1.) * delta_q_dofs / 1.e6 /
          (current_.cpu_time_sum - previous_.cpu_time_sum);
      data.fudged_cpu_timings = fudge_cpu_timings;

      /* avoid printing small negative numbers: */
      data.cpu_time_skew =
          std::max(0.,
                   (current_.cpu_time_max - current_.cpu_time_min - //
                    previous_.cpu_time_max + previous_.cpu_time_min) /
                       delta_cycles);
      data.cpu_time_skew_fraction =
          data.cpu_time_skew * delta_cycles /
          (current_.cpu_time_avg - previous_.cpu_time_avg);
    }

    const auto &hyperbolic_module = simulation_.hyperbolic_module();
    const auto &parabolic_module = simulation_.parabolic_module();
    data.time_stepping_scheme =
        simulation_.time_integrator().time_stepping_scheme_name();
    data.cfl = hyperbolic_module.cfl();
    data.restarts = {hyperbolic_module.n_restarts(),
                     parabolic_module.n_restarts()};
    data.warnings = {hyperbolic_module.n_warnings(),
                     parabolic_module.n_warnings()};
    data.corrections = {hyperbolic_module.n_corrections(),
                        parabolic_module.n_corrections()};

    data.solver_statistics.clear();
    if (!simulation_.parabolic_system().is_identity()) {
      std::ostringstream solver_statistics;
      parabolic_module.print_solver_statistics(solver_statistics);

      /* Strip the indentation and brackets of every line: */
      std::istringstream lines(solver_statistics.str());
      for (std::string line; std::getline(lines, line);) {
        const auto first = line.find_first_not_of(" [");
        const auto last = line.find_last_not_of(" ]");
        if (first != std::string::npos)
          data.solver_statistics.push_back(
              line.substr(first, last - first + 1));
      }
    }

    /* And estimate the remaining wall time: */

    time_per_second_exp_ =
        0.8 * time_per_second_exp_ + 0.2 * data.time_per_second;
    data.eta = std::max(context.t_final - context.t, Number(0.)) /
               time_per_second_exp_;
  }
} // namespace ryujin
