//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <deal.II/base/mpi.h>

#include <array>
#include <deque>
#include <string>
#include <vector>

namespace ryujin
{
  /**
   * A snapshot of the cycle statistics gathered by the Logger on rank 0,
   * shown on the terminal dashboard or rendered as plain text with
   * render_text().
   *
   * @ingroup Driver
   */
  struct CycleStatistics {
    using MinMaxAvg = dealii::Utilities::MPI::MinMaxAvg;

    /**
     * Statistics of a single ComputingTimer.
     */
    struct Timer {
      /** The name of the timer. */
      std::string name;
      /** The wall time over all ranks. */
      MinMaxAvg wall_time = {};
      /** The cpu time summed over all ranks. */
      double cpu_time = 0.;
      /** The fraction of the total cpu time, negative if not meaningful. */
      double fraction = -1.;
    };

    /**
     * Status of an active task and its trigger.
     */
    struct Task {
      /** The name of the task. */
      std::string name;
      /** The phase of the time loop the task runs in. */
      std::string phase;
      /** The condition of the trigger, empty if the task has none. */
      std::string trigger;
      /** When the trigger fired last, empty if it has not fired yet. */
      std::string last;
      /** When the trigger fires next, empty if it does not fire again. */
      std::string next;
      /** The progress towards the next firing, negative if not meaningful. */
      double progress = -1.;
    };

    /**
     * A status message and the cycle and time at which it was issued.
     */
    struct Message {
      /** The text of the message. */
      std::string text;
      /** The cycle at which the message was issued. */
      unsigned int cycle = 0;
      /** The simulation time at which the message was issued. */
      double t = 0.;
    };

    /**
     * @name Header
     */
    //@{

    /** Whether this is the final snapshot of the run. */
    bool final_time = false;
    /** The current cycle. */
    unsigned int cycle = 0;
    /** The current simulation time. */
    double t = 0.;
    /** The final simulation time. */
    double t_final = 0.;
    /** The estimated remaining wall time in seconds. */
    double eta = 0.;
    /** The wall time in seconds between two snapshots. */
    double terminal_update_interval = 0.;

    //@}
    /**
     * @name Information
     */
    //@{

    /** The problem name of the hyperbolic system. */
    std::string hyperbolic_name;
    /** The problem name of the parabolic system, empty for the identity. */
    std::string parabolic_name;
    /** The base name of all output files. */
    std::string base_name;
    /** The name of the Kokkos execution backend. */
    std::string backend;
    /** A description of the threads in use, such as "8 threads". */
    std::string threads;
    /** The number of global degrees of freedom. */
    double n_global_dofs = 0.;
    /** The number of ensembles. */
    unsigned int n_ensembles = 1;
    /** The number of MPI ranks over all ensembles. */
    unsigned int n_ranks = 1;

    //@}
    /**
     * @name Messages
     */
    //@{

    /** The most recent status messages. */
    std::deque<Message> messages;

    //@}
    /**
     * @name Memory, timers, and tasks
     */
    //@{

    /** The resident memory of all ranks in MiB. */
    MinMaxAvg memory = {};
    /** The statistics of all timers. */
    std::vector<Timer> timers;
    /** The status of all active tasks. */
    std::vector<Task> tasks;

    //@}
    /**
     * @name Throughput
     */
    //@{

    /** Whether ryujin runs on a device with a separate memory space. */
    bool device = false;
    /** Whether the cpu timings are doubled to correct for hyperthreading. */
    bool fudged_cpu_timings = false;
    /** The number of devices. */
    unsigned int n_devices = 0;
    /** The throughput in million dofs per second of compute time. */
    double compute_m_dofs_per_sec = 0.;
    /** The throughput in million dofs per second of wall time. */
    double wall_m_dofs_per_sec = 0.;
    /** The number of cycles per second of wall time. */
    double cycles_per_second = 0.;
    /** The fraction of the wall time the devices are busy. */
    double utilization = 0.;
    /** The device time since the last snapshot. */
    double delta_device_time = 0.;
    /** The wall time since the last snapshot. */
    double delta_wall_time = 0.;
    /** The skew of the cpu time over all ranks per cycle. */
    double cpu_time_skew = 0.;
    /** The cpu time skew relative to the average cpu time per cycle. */
    double cpu_time_skew_fraction = 0.;
    /** The skew of the device time over all ranks per cycle. */
    double device_time_skew = 0.;
    /** The device time skew relative to the average device time per cycle. */
    double device_time_skew_fraction = 0.;
    /** The rank with the smallest accumulated device time. */
    unsigned int device_time_min_index = 0;
    /** The rank with the largest accumulated device time. */
    unsigned int device_time_max_index = 0;

    /** The name of the time stepping scheme. */
    std::string time_stepping_scheme;
    /** The current CFL number. */
    double cfl = 0.;
    /** The restarts of the hyperbolic and the parabolic module. */
    std::array<unsigned int, 2> restarts = {};
    /** The warnings of the hyperbolic and the parabolic module. */
    std::array<unsigned int, 2> warnings = {};
    /** The statistics of the linear solvers of the parabolic module. */
    std::vector<std::string> solver_statistics;
    /** The average time step size since the last snapshot. */
    double tau = 0.;
    /** The simulation time advanced per second of wall time. */
    double time_per_second = 0.;

    //@}
  };


  /**
   * Render the snapshot @p data as plain text, for the log file and
   * for terminals without the dashboard.
   *
   * @ingroup Driver
   */
  std::string render_text(const CycleStatistics &data);
} // namespace ryujin
