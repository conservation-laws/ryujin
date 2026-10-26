//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/interface/task.h>

#include <deal.II/base/mpi_stub.h>
#include <deal.II/base/parameter_acceptor.h>

#include <functional>
#include <string>
#include <vector>

namespace ryujin
{
  /**
   * The high-level time loop driving the computation.
   *
   * The Scheduler runs tasks (see Interface::Task) in cycles.
   *
   * Every cycle consists of four phases:
   *  - TaskPhase::pre_step: output and postprocessing at time t,
   *  - TaskPhase::adapt: mesh adaptation at time t (only if a time step
   *    follows),
   *  - TaskPhase::step: the time step from t to t + tau, and
   *  - TaskPhase::post_step: tasks at time t + tau.
   *
   * The active tasks are the required tasks (in the given order),
   * followed by the selectable tasks chosen with the run time parameter
   * "tasks" (in the order of the parameter). In every phase the Scheduler
   * executes the active tasks of the phase in this order. The first
   * required task is the primary task: it sets up the discretization and
   * chooses the time step size.
   *
   * @ingroup Driver
   */
  template <typename Number = double>
  class Scheduler final : public dealii::ParameterAcceptor
  {
  public:
    using Task = Interface::Task<Number>;

    /**
     * Constructor. The Scheduler runs the @p required tasks and the
     * selectable @p tasks chosen with the run time parameter "tasks". The
     * status messages of the Scheduler and all tasks are passed to @p
     * info; by default they are printed on rank 0 of @p mpi_comm.
     */
    Scheduler(const MPI_Comm &mpi_comm,
              const std::vector<Task *> &required,
              const std::vector<Task *> &tasks = {},
              const std::function<void(const std::string &)> &info = {});

    /**
     * Run the high-level time loop.
     */
    void run();

  private:
    /**
     * @name Run time options
     */
    //@{

    /** The base name of all output files. */
    std::string base_name_;

    /** The final simulation time. */
    Number t_final_;
    /** The maximal number of cycles, zero for no limit. */
    unsigned int max_cycles_;
    /** Shorten time steps to reach every timer tick exactly. */
    bool align_time_steps_;
    /** The simulation time between two timer ticks. */
    Number timer_granularity_;

    /** The names of the selectable tasks to run, in order. */
    std::vector<std::string> task_names_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    /** The MPI communicator. */
    const MPI_Comm mpi_comm_;

    /** The tasks that always run, the primary task first. */
    const std::vector<Task *> required_;
    /** The tasks that can be chosen with the parameter "tasks". */
    const std::vector<Task *> selectable_;

    /** The printer of the status messages. */
    std::function<void(const std::string &)> info_;

    /** The active tasks in the order of execution. */
    std::vector<Task *> tasks_;

    //@}
    /**
     * @name Internal methods
     */
    //@{

    /**
     * Run all active tasks of the given @p phase.
     */
    void run_phase(TaskPhase phase, TaskContext<Number> &context);

    //@}
  };

} // namespace ryujin
