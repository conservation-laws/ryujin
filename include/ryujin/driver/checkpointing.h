//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/interface/simulation.h>
#include <ryujin/interface/task.h>

#include <deal.II/base/parameter_acceptor.h>

#include <string>

namespace ryujin
{
  /**
   * The task "checkpointing" that writes checkpoints of the computation
   * whenever its trigger fires (by default every hour of wall time) and
   * at the end, and that resumes an interrupted computation from the
   * last checkpoint.
   *
   * A checkpoint consists of the mesh with the attached state vector of
   * the Simulation, the current time and timer cycle, and the internal
   * state of all active tasks (see Interface::Task::save()).
   *
   * @ingroup Driver
   */
  template <int dim, typename Number = double>
  class Checkpointing final : public dealii::ParameterAcceptor,
                              public Interface::Task<Number>
  {
  public:
    /**
     * Constructor.
     */
    Checkpointing(Interface::Simulation<dim, Number> &simulation,
                  const std::string &subsection = "/Checkpointing");

    /**
     * @name Task
     */
    //@{

    /**
     * Resume from the last checkpoint (if "resume" is set): load the
     * mesh with the attached state vector, restore the current time and
     * timer cycle, and the internal state of all active tasks.
     */
    void start(TaskContext<Number> &context) override;

    /**
     * Write a checkpoint whenever the trigger fires.
     */
    void execute(TaskContext<Number> &context) override;

    /**
     * Write a final checkpoint.
     */
    void finalize(TaskContext<Number> &context) override;

    //@}

  private:
    /**
     * @name Run time options
     */
    //@{

    bool resume_;
    bool resume_at_time_zero_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    Trigger<Number> trigger_;

    Interface::Simulation<dim, Number> &simulation_;

    //@}
    /**
     * @name Internal methods
     */
    //@{

    /**
     * Write out a checkpoint to disk.
     */
    void write_checkpoint(const TaskContext<Number> &context);

    //@}
  };
} // namespace ryujin
