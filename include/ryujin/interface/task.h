//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/patterns_conversion.h>

#include <deal.II/base/parameter_acceptor.h>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace ryujin
{
  /**
   * The phase of a cycle of the Scheduler in which a Task runs.
   *
   * @ingroup Driver
   */
  enum class TaskPhase : std::uint8_t {
    /**
     * Before the time step, at time t. Output and postprocessing tasks
     * typically run in this phase.
     */
    pre_step,

    /**
     * Mesh adaptation, at time t. The Scheduler runs this phase only if a
     * time step follows. A task of this phase that changes the mesh sets
     * TaskContext::mesh_changed.
     */
    adapt,

    /**
     * The time step from t to t + tau. The first task of this phase
     * chooses the time step size tau, all subsequent tasks advance over
     * the same interval (Lie splitting).
     */
    step,

    /**
     * After the time step, at time t + tau.
     */
    post_step,
  };


  /**
   * The condition under which a Trigger fires.
   *
   * @ingroup Driver
   */
  enum class TriggerMode : std::uint8_t {
    /**
     * Fire in every cycle whose number is a multiple of the "multiplier".
     */
    every_cycle,

    /**
     * Fire whenever a timer tick is reached and the timer cycle is a
     * multiple of the "multiplier".
     */
    timer_tick,

    /**
     * Fire whenever the wall time has advanced by more than the
     * "interval" (in seconds) since the last time the trigger fired.
     */
    wall_clock,

    /**
     * Fire once whenever the current time has reached one of the "time
     * points". Time points that lie before the time at which the trigger
     * is evaluated for the first time are ignored.
     */
    time_points,
  };
} // namespace ryujin

#ifndef DOXYGEN
DECLARE_ENUM(ryujin::TriggerMode,
             LIST({ryujin::TriggerMode::every_cycle, "every cycle"},
                  {ryujin::TriggerMode::timer_tick, "timer tick"},
                  {ryujin::TriggerMode::wall_clock, "wall clock"},
                  {ryujin::TriggerMode::time_points, "time points"}, ));
#endif

namespace ryujin
{
  namespace Interface
  {
    template <typename Number>
    class Task;
  } // namespace Interface


  /**
   * The context that the Scheduler shares with all tasks.
   *
   * All values are identical on all MPI ranks so that tasks can base
   * collective decisions on them.
   *
   * @ingroup Driver
   */
  template <typename Number = double>
  struct TaskContext {
    /**
     * Return whether the current time has reached the timer tick @p
     * timer_cycle.
     */
    bool tick_reached() const
    {
      constexpr Number relax =
          Number(1.) - Number(10.) * std::numeric_limits<Number>::epsilon();
      return t >= relax * timer_cycle * timer_granularity;
    }

    /**
     * The current time.
     */
    Number t = Number(0.);

    /**
     * The time step size. It is zero at the begin of the step phase and
     * set by the first task of the step phase.
     */
    Number tau = Number(0.);

    /**
     * The largest time that the time step must not exceed. Valid in the
     * step phase.
     */
    Number t_max = std::numeric_limits<Number>::max();

    /**
     * The current cycle.
     */
    unsigned int cycle = 0;

    /**
     * The current timer cycle, i.e., the index of the next timer tick.
     */
    unsigned int timer_cycle = 0;

    /**
     * Whether the current time has reached the timer tick @p
     * timer_cycle, see tick_reached(). The Scheduler sets it at the begin
     * of every phase.
     */
    bool timer_tick = false;

    /**
     * The wall time of the time loop, synchronized over all MPI ranks
     * after every time step.
     */
    Number wall_time = Number(0.);

    /**
     * The final time.
     */
    Number t_final = Number(0.);

    /**
     * The time interval between two timer ticks.
     */
    Number timer_granularity = Number(0.);

    /**
     * The base name of all output files.
     */
    std::string base_name;

    /**
     * Set by a task of the adapt phase that has changed the mesh. The
     * Scheduler then prepares all tasks for the new mesh.
     */
    bool mesh_changed = false;

    /**
     * All active tasks in the order of execution.
     */
    std::span<Interface::Task<Number> *const> tasks;

    /**
     * Print a status message.
     */
    std::function<void(const std::string &)> info;
  };


  /**
   * A Trigger decides in which cycles a task performs (part of) its work:
   * every n-th cycle, at every n-th timer tick, after an interval of wall
   * time, or at given time points.
   *
   * A task owns a Trigger for every cadence it needs and evaluates it in
   * Interface::Task::execute() with fire().
   *
   * @ingroup Driver
   */
  template <typename Number = double>
  class Trigger
  {
  public:
    /**
     * Constructor.
     */
    Trigger(TriggerMode mode = TriggerMode::timer_tick,
            unsigned int multiplier = 1,
            Number interval = Number(0.))
        : mode_(mode)
        , multiplier_(multiplier)
        , interval_(interval)
        , last_fired_(Number(0.))
    {
    }

    Trigger(const Trigger &) = delete;
    Trigger &operator=(const Trigger &) = delete;

    /**
     * Declare the run time parameters "<prefix>", "<prefix> multiplier",
     * "<prefix> interval", and "<prefix> time points" in the subsection
     * of @p acceptor.
     */
    void add_parameters(dealii::ParameterAcceptor &acceptor,
                        const std::string &prefix = "trigger")
    {
      acceptor.add_parameter(
          prefix,
          mode_,
          "The condition under which the task is executed: every cycle, timer "
          "tick, wall clock, time points");

      acceptor.add_parameter(prefix + " multiplier",
                             multiplier_,
                             "Execute only in every n-th cycle (every cycle), "
                             "or at every n-th timer tick (timer tick)");

      acceptor.add_parameter(prefix + " interval",
                             interval_,
                             "Execute after the given number of seconds of "
                             "wall time have passed (wall clock)");

      acceptor.add_parameter(prefix + " time points",
                             time_points_,
                             "Execute once whenever the time has reached one "
                             "of the given time points (time points)");

      acceptor.parse_parameters_call_back.connect([this] {
        std::ranges::sort(time_points_);
        next_time_point_ = std::numeric_limits<std::size_t>::max();
      });
    }

    /**
     * Return whether the trigger fires in the current cycle.
     */
    bool fire(const TaskContext<Number> &context)
    {
      switch (mode_) {
      case TriggerMode::every_cycle:
        return context.cycle % multiplier_ == 0;

      case TriggerMode::timer_tick:
        return context.timer_tick && context.timer_cycle % multiplier_ == 0;

      case TriggerMode::wall_clock:
        if (context.wall_time < last_fired_ + interval_)
          return false;
        last_fired_ = context.wall_time;
        return true;

      case TriggerMode::time_points: {
        /* Skip all time points before the first evaluation: */
        if (next_time_point_ == std::numeric_limits<std::size_t>::max()) {
          next_time_point_ = 0;
          while (next_time_point_ < time_points_.size() &&
                 time_points_[next_time_point_] < context.t)
            ++next_time_point_;
        }

        bool fire = false;
        while (next_time_point_ < time_points_.size() &&
               time_points_[next_time_point_] <= context.t) {
          ++next_time_point_;
          fire = true;
        }
        return fire;
      }
      }

      __builtin_unreachable();
    }

  private:
    TriggerMode mode_;
    unsigned int multiplier_;
    Number interval_;
    std::vector<Number> time_points_;

    Number last_fired_;
    std::size_t next_time_point_ = std::numeric_limits<std::size_t>::max();
  };


  namespace Interface
  {
    /**
     * The interface of a Task: an operation that the Scheduler executes in
     * a given TaskPhase of every cycle.
     *
     * The Scheduler calls the functions of all active tasks in the
     * following order:
     *  - initialize() once at the begin,
     *  - start() once after all tasks have been initialized,
     *  - prepare() after start() and after every mesh adaptation,
     *  - max_tau() before, and execute() in the phase of the task in
     *    every cycle,
     *  - finalize() once at the end, in reverse order.
     *
     * In addition, a task stores its internal state in checkpoints with
     * save() and load().
     *
     * @ingroup Interfaces
     */
    template <typename Number = double>
    class Task
    {
    public:
      /**
       * Destructor.
       */
      virtual ~Task() = default;

      /**
       * @name Setup
       */
      //@{

      /**
       * Initialize the task. Called once at the begin.
       */
      virtual void initialize(TaskContext<Number> & /*context*/) {}

      /**
       * Start the computation. Called once after all tasks have been
       * initialized and before the first call to prepare().
       */
      virtual void start(TaskContext<Number> & /*context*/) {}

      /**
       * Prepare the task for the current mesh. Called after start() and
       * after every mesh adaptation.
       *
       * @note prepare() must preserve the internal state of the task.
       */
      virtual void prepare(TaskContext<Number> & /*context*/) {}

      //@}
      /**
       * @name Execution
       */
      //@{

      /**
       * Return the largest time step size that the task admits at the
       * current state.
       */
      virtual Number max_tau(const TaskContext<Number> & /*context*/) const
      {
        return std::numeric_limits<Number>::max();
      }

      /**
       * Execute the task. Called in every cycle in the phase of the task.
       */
      virtual void execute(TaskContext<Number> &context) = 0;

      /**
       * Finalize the task. Called once at the end.
       */
      virtual void finalize(TaskContext<Number> & /*context*/) {}

      //@}
      /**
       * @name Checkpointing
       */
      //@{

      /**
       * Store the internal state of the task in a checkpoint. Called on
       * all MPI ranks; the data of rank 0 is written.
       */
      virtual void save(boost::archive::binary_oarchive & /*archive*/) const {}

      /**
       * Restore the internal state of the task from a checkpoint. Called
       * on all MPI ranks with the data written by save() on rank 0.
       */
      virtual void load(boost::archive::binary_iarchive & /*archive*/) {}

      //@}
      /**
       * @name Information
       */
      //@{

      /**
       * The name of the task, used for selecting it in the parameter
       * file.
       */
      ACCESSOR_READ_ONLY(name)

      /**
       * The phase in which the task runs.
       */
      ACCESSOR_READ_ONLY(phase)

      //@}

    protected:
      /**
       * Constructor taking the name of the task and the phase in which it
       * runs.
       */
      Task(const std::string &name, TaskPhase phase)
          : name_(name)
          , phase_(phase)
      {
      }

    private:
      std::string name_;
      TaskPhase phase_;
    };
  } // namespace Interface
} // namespace ryujin
