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

    Number terminal_update_interval_;
    bool terminal_correct_for_hyperthreading_;
    DashboardSettings dashboard_settings_;

    std::string debug_command_;
    std::string debug_filename_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const Interface::Simulation<dim, Number> &simulation_;
    const MPIEnsemble &mpi_ensemble_;

    dealii::types::global_dof_index n_global_dofs_;
    unsigned int n_devices_;

    std::ofstream logfile_;

    Number last_terminal_output_;

    struct ThroughputData {
      unsigned int cycle = 0;
      double t = 0.;
      double cpu_time_sum = 0.;
      double cpu_time_avg = 0.;
      double cpu_time_min = 0.;
      double cpu_time_max = 0.;
      double wall_time = 0.;
      double device_time_sum = 0.;
    } previous_, current_;

    double time_per_second_exp_;

    DashboardData dashboard_data_;
#if defined(WITH_FTXUI)
    Dashboard dashboard_;
#endif

    /* The original status message printer of the TaskContext: */
    std::function<void(const std::string &)> print_info_;
    /* Whether status messages go to the dashboard instead of stdout: */
    bool dashboard_active_ = false;

    //@}
    /**
     * @name Internal methods
     */
    //@{

    bool interactive_terminal() const;

    void print_debug_output();

    void print_parameters(const std::string &base_name, std::ostream &stream);

    void print_mpi_partition(std::ostream &stream);

    void print_device_information();

    void print_head(const std::string &header,
                    const std::string &secondary,
                    std::ostream &stream);

    void print_information(const TaskContext<Number> &context,
                           std::ostream &stream);

    void print_memory_statistics(std::ostream &stream);

    void print_timers(std::ostream &stream);

    void print_tasks(const TaskContext<Number> &context, std::ostream &stream);

    void print_throughput(const TaskContext<Number> &context,
                          std::ostream &stream,
                          bool final_time = false);

    void print_cycle_statistics(const TaskContext<Number> &context,
                                bool write_to_logfile = false,
                                bool final_time = false);

    //@}
  };
} // namespace ryujin
