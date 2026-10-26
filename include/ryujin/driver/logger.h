//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/driver/dashboard.h>
#include <ryujin/interface/simulation.h>
#include <ryujin/interface/task.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/types.h>

#include <fstream>
#include <functional>
#include <ostream>
#include <string>

namespace ryujin
{
  /**
   * The task "logger" that records the run time parameters, the MPI
   * partition, and the device information in the log file
   * "<basename>.log", and prints cycle statistics to the terminal and
   * the log file.
   *
   * @ingroup Driver
   */
  template <int dim, typename Number = double>
  class Logger final : public dealii::ParameterAcceptor,
                       public Interface::Task<Number>
  {
  public:
    /**
     * Constructor.
     */
    Logger(const Interface::Simulation<dim, Number> &simulation,
           const std::string &subsection = "/Logger");

    /**
     * @name Task
     */
    //@{

    /**
     * Open the log file and record the run time parameters.
     */
    void initialize(TaskContext<Number> &context) override;

    /**
     * Record the device information.
     */
    void start(TaskContext<Number> &context) override;

    /**
     * Record the MPI partition of the current mesh. The first call starts
     * the animation of the dashboard.
     */
    void prepare(TaskContext<Number> &context) override;

    /**
     * Print cycle statistics to the terminal (every "terminal update
     * interval" seconds) and the log file (at every timer tick).
     */
    void execute(TaskContext<Number> &context) override;

    /**
     * Print the final cycle statistics and the debug output.
     */
    void finalize(TaskContext<Number> &context) override;

    //@}

  private:
    /**
     * @name Run time options
     */
    //@{

    /** The wall time in seconds between two terminal updates. */
    Number terminal_update_interval_;
    /** Halve the cpu time if there are two threads per rank. */
    bool terminal_correct_for_hyperthreading_;
    /** The settings of the terminal dashboard. */
    DashboardSettings dashboard_settings_;

    /** A shell command run at the end, if nonempty. */
    std::string debug_command_;
    /** A file printed at the end, if nonempty. */
    std::string debug_filename_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    /** The simulation whose statistics we record. */
    const Interface::Simulation<dim, Number> &simulation_;
    /** The MPI ensemble of the simulation. */
    const MPIEnsemble &mpi_ensemble_;

    /** The number of global degrees of freedom summed over all ensembles. */
    dealii::types::global_dof_index n_global_dofs_;
    /** The number of distinct devices of all ranks. */
    unsigned int n_devices_;

    /** The log file "<basename>.log", open on rank 0 only. */
    std::ofstream logfile_;

    /** The wall time of the last terminal update. */
    Number last_terminal_output_;

    /**
     * The accumulated times of the time loop at a snapshot, summed,
     * averaged, and minimized and maximized over all ranks.
     */
    struct ThroughputData {
      /** The cycle of the snapshot. */
      unsigned int cycle = 0;
      /** The simulation time of the snapshot. */
      double t = 0.;
      /** The cpu time summed over all ranks. */
      double cpu_time_sum = 0.;
      /** The average cpu time of a rank. */
      double cpu_time_avg = 0.;
      /** The smallest cpu time of a rank. */
      double cpu_time_min = 0.;
      /** The largest cpu time of a rank. */
      double cpu_time_max = 0.;
      /** The largest wall time of a rank. */
      double wall_time = 0.;
      /** The device time summed over all ranks. */
      double device_time_sum = 0.;
      /** The average device time of a rank. */
      double device_time_avg = 0.;
      /** The smallest device time of a rank. */
      double device_time_min = 0.;
      /** The largest device time of a rank. */
      double device_time_max = 0.;
    } previous_, current_;

    /** A moving average of the simulation time per second of wall time. */
    double time_per_second_exp_;

    /** The snapshot of the cycle statistics shown on the terminal. */
    DashboardData dashboard_data_;
#if defined(WITH_FTXUI)
    /** The terminal dashboard. */
    Dashboard dashboard_;
#endif

    /** The original status message printer of the TaskContext. */
    std::function<void(const std::string &)> print_info_;
    /** Whether status messages go to the dashboard instead of stdout. */
    bool dashboard_active_ = false;

    //@}
    /**
     * @name Internal methods
     */
    //@{

    /** Return whether we draw the dashboard instead of plain text. */
    bool interactive_terminal() const;

    /** Run the debug command and print the debug file. */
    void print_debug_output();

    /** Print the revision and all run time parameters to @p stream. */
    void print_parameters(const std::string &base_name, std::ostream &stream);

    /** Print the statistics of the MPI partition to @p stream. */
    void print_mpi_partition(std::ostream &stream);

    /** Count the devices and record their information in the log file. */
    void print_device_information();

    /** Collect a snapshot and print it to the terminal and the log file. */
    void print_cycle_statistics(const TaskContext<Number> &context,
                                bool write_to_logfile = false,
                                bool final_time = false);

    /** Collect the header and the information of the snapshot. */
    void collect_information(const TaskContext<Number> &context);

    /** Collect the resident memory of all ranks. */
    void collect_memory_statistics();

    /** Collect the statistics of all timers. */
    void collect_timers();

    /** Collect the status of all active tasks and their triggers. */
    void collect_tasks(const TaskContext<Number> &context);

    /** Collect the throughput since the last snapshot. */
    void collect_throughput(const TaskContext<Number> &context,
                            bool final_time);

    //@}
  };
} // namespace ryujin
