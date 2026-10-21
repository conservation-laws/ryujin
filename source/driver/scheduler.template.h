//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/computing_timer.h>
#include <ryujin/driver/scheduler.h>

#include <deal.II/base/mpi.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <ranges>

namespace ryujin
{
  template <typename Number>
  Scheduler<Number>::Scheduler(
      const MPI_Comm &mpi_comm,
      const std::vector<Task *> &required,
      const std::vector<Task *> &tasks /*= {}*/,
      const std::function<void(const std::string &)> &info /*= {}*/)
      : ParameterAcceptor("/A - Scheduler")
      , mpi_comm_(mpi_comm)
      , required_(required)
      , selectable_(tasks)
      , info_(info)
  {
    AssertThrow(!required_.empty(),
                dealii::ExcMessage("The Scheduler needs a primary task."));

    if (!info_)
      info_ = [this](const std::string &message) {
        if (dealii::Utilities::MPI::this_mpi_process(mpi_comm_) == 0)
          std::cout << "[INFO] " << message << std::endl;
      };

    base_name_ = "test";
    add_parameter("basename", base_name_, "Base name for all output files");

    t_final_ = Number(5.);
    add_parameter("final time", t_final_, "Final time");

    align_time_steps_ = false;
    add_parameter("align time steps",
                  align_time_steps_,
                  "If set to true, time steps are shortened so that every "
                  "timer tick and the final time are reached exactly");

    timer_granularity_ = Number(0.01);
    add_parameter("timer granularity",
                  timer_granularity_,
                  "The timer granularity specifies the time interval of a "
                  "timer tick. Tasks with trigger \"timer tick\" are executed "
                  "at every n-th timer tick, where n is the \"trigger "
                  "multiplier\" of the task");

    std::string names;
    for (const auto *task : selectable_)
      names += (names.empty() ? "" : "|") + task->name();

    add_parameter(
        "tasks",
        task_names_,
        "Ordered list of tasks that are executed in addition to the "
        "required tasks. Within every phase of a cycle tasks are executed "
        "in the given order.",
        this->prm,
        dealii::Patterns::List(dealii::Patterns::Selection(names),
                               0,
                               dealii::Patterns::List::max_int_value,
                               ","));
  }


  template <typename Number>
  void Scheduler<Number>::run()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Scheduler<Number>::run()" << std::endl;
#endif

    ComputingTimer::reinit();

    /* Select the active tasks: the required tasks and the selected ones */

    tasks_ = required_;
    for (const auto &name : task_names_) {
      const auto it = std::ranges::find_if(
          selectable_, [&](const auto *task) { return task->name() == name; });
      AssertThrow(it != selectable_.end(),
                  dealii::ExcMessage("There is no task »" + name + "«."));
      AssertThrow(std::ranges::find(tasks_, *it) == tasks_.end(),
                  dealii::ExcMessage("The task »" + name +
                                     "« is selected more than once."));
      tasks_.push_back(*it);
    }

    TaskContext<Number> context{.t_final = t_final_,
                                .timer_granularity = timer_granularity_,
                                .base_name = base_name_,
                                .tasks = tasks_,
                                .info = info_};

    /*
     * Initialize and prepare all tasks:
     */

    for (auto *task : tasks_)
      task->initialize(context);

    {
      ComputingTimer::Scope scope("(re)initialize data structures");
      context.info("initializing data structures");

      for (auto *task : tasks_)
        task->start(context);

      for (auto *task : tasks_)
        task->prepare(context);
    }

    /*
     * The honorable main loop:
     */

    context.info("entering main loop");
    ComputingTimer::timer("time loop").start();

    constexpr Number relax =
        Number(1.) - Number(10.) * std::numeric_limits<Number>::epsilon();

    for (context.cycle = 1;; ++context.cycle) {

#ifdef DEBUG_OUTPUT
      std::cout << "\n\n###   cycle = " << context.cycle << "   ###\n\n"
                << std::endl;
#endif

      run_phase(TaskPhase::pre_step, context);

      if (context.timer_tick)
        ++context.timer_cycle;

      /* Break if we have reached the final time. */

      if (context.t >= relax * t_final_)
        break;

      /* Adapt the mesh and prepare all tasks if requested: */

      context.prepare_required = false;
      run_phase(TaskPhase::adapt, context);

      if (context.prepare_required) {
        ComputingTimer::Scope scope("(re)initialize data structures");

        for (auto *task : tasks_)
          task->prepare(context);
      }

      /* Perform a time step: */

      context.tau = Number(0.);
      context.t_max =
          align_time_steps_
              ? std::min(t_final_, context.timer_cycle * timer_granularity_)
              : std::numeric_limits<Number>::max();

      run_phase(TaskPhase::step, context);

      AssertThrow(context.tau > Number(0.),
                  dealii::ExcMessage("No task of the step phase has chosen "
                                     "a time step size."));
      context.t += context.tau;

      /* Synchronize wall time: */

      context.wall_time = ComputingTimer::timer("time loop").wall_time();
      {
        ComputingTimer::Scope scope(
            "time step [X] _ - synchronization barriers");
        context.wall_time =
            dealii::Utilities::MPI::max(context.wall_time, mpi_comm_);
      }

      run_phase(TaskPhase::post_step, context);
    }

    /* We have actually performed one cycle less. */
    --context.cycle;

    ComputingTimer::timer("time loop").stop();

    for (auto *task : tasks_ | std::views::reverse)
      task->finalize(context);
  }


  template <typename Number>
  void Scheduler<Number>::run_phase(const TaskPhase phase,
                                    TaskContext<Number> &context)
  {
    context.timer_tick = context.tick_reached();

    for (auto *task : tasks_)
      if (task->phase() == phase)
        task->execute(context);
  }
} // namespace ryujin
