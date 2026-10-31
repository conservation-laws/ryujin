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
#include <memory>
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
     * TaskContext::prepare_required.
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
     * Return whether the current time has reached the time @p time up to
     * a relative tolerance of a few machine epsilon.
     */
    bool time_reached(const Number time) const
    {
      constexpr Number relax =
          Number(1.) - Number(10.) * std::numeric_limits<Number>::epsilon();
      return t >= relax * time;
    }

    /**
     * Return whether the current time has reached the timer tick @p
     * timer_cycle.
     */
    bool tick_reached() const
    {
      return time_reached(timer_cycle * timer_granularity);
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
     * Set by a task of the adapt phase that has invalidated the data
     * structures of the tasks, for example by changing the mesh. The
     * Scheduler then calls Interface::Task::prepare() on all tasks.
     */
    bool prepare_required = false;

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
      if (!evaluate(context))
        return false;

      fired_ = true;
      last_cycle_ = context.cycle;
      last_t_ = context.t;
      last_wall_time_ = context.wall_time;
      return true;
    }

    /**
     * Return when the trigger fires next: the cycle (every cycle), the
     * time (timer tick, time points), or the wall time (wall clock).
     * Returns std::numeric_limits<Number>::max() if the trigger does not
     * fire again.
     */
    Number next(const TaskContext<Number> &context) const
    {
      constexpr auto never = std::numeric_limits<Number>::max();

      switch (mode_) {
      case TriggerMode::every_cycle:
        return (context.cycle / multiplier_ + 1) * multiplier_;

      case TriggerMode::timer_tick: {
        /* The smallest multiple of the multiplier not below timer_cycle: */
        const auto k =
            (context.timer_cycle + multiplier_ - 1) / multiplier_ * multiplier_;
        const Number t = k * context.timer_granularity;
        return t <= context.t_final ? t : never;
      }

      case TriggerMode::wall_clock:
        return last_wall_time_ + interval_;

      case TriggerMode::time_points: {
        const auto it =
            next_time_point_ == std::numeric_limits<std::size_t>::max()
                ? std::ranges::lower_bound(time_points_, context.t)
                : time_points_.begin() + next_time_point_;
        return it != time_points_.end() ? *it : never;
      }
      }

      __builtin_unreachable();
    }

    /**
     * Return how far the trigger has advanced from the previous to the
     * next firing as a fraction in [0, 1], or a negative value if the
     * trigger does not fire again.
     */
    double progress(const TaskContext<Number> &context) const
    {
      const Number next_firing = next(context);
      if (next_firing == std::numeric_limits<Number>::max())
        return -1.;

      const auto fraction = [](double begin, double now, double end) {
        return end > begin ? std::clamp((now - begin) / (end - begin), 0., 1.)
                           : 1.;
      };

      switch (mode_) {
      case TriggerMode::every_cycle:
        return fraction(next_firing - multiplier_, context.cycle, next_firing);

      case TriggerMode::timer_tick:
        return fraction(next_firing - multiplier_ * context.timer_granularity,
                        context.t,
                        next_firing);

      case TriggerMode::wall_clock:
        return fraction(last_wall_time_, context.wall_time, next_firing);

      case TriggerMode::time_points: {
        const auto it = std::ranges::lower_bound(time_points_, next_firing);
        const Number previous =
            it != time_points_.begin() ? *(it - 1) : Number(0.);
        return fraction(previous, context.t, next_firing);
      }
      }

      __builtin_unreachable();
    }

    /**
     * The condition under which the trigger fires.
     */
    ACCESSOR_READ_ONLY(mode)

    /**
     * The multiplier of the modes "every cycle" and "timer tick".
     */
    ACCESSOR_READ_ONLY(multiplier)

    /**
     * The interval of the mode "wall clock" (in seconds).
     */
    ACCESSOR_READ_ONLY(interval)

    /**
     * The time points of the mode "time points".
     */
    ACCESSOR_READ_ONLY(time_points)

    /**
     * Whether the trigger has fired at least once.
     */
    ACCESSOR_READ_ONLY(fired)

    /**
     * The cycle, the time, and the wall time at which the trigger fired
     * last.
     */
    ACCESSOR_READ_ONLY(last_cycle)
    ACCESSOR_READ_ONLY(last_t)
    ACCESSOR_READ_ONLY(last_wall_time)

  private:
    /**
     * Return whether the condition of the trigger is met in the current
     * cycle.
     */
    bool evaluate(const TaskContext<Number> &context)
    {
      switch (mode_) {
      case TriggerMode::every_cycle:
        return context.cycle % multiplier_ == 0;

      case TriggerMode::timer_tick:
        return context.timer_tick && context.timer_cycle % multiplier_ == 0;

      case TriggerMode::wall_clock:
        return context.wall_time >= last_wall_time_ + interval_;

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
               context.time_reached(time_points_[next_time_point_])) {
          ++next_time_point_;
          fire = true;
        }
        return fire;
      }
      }

      __builtin_unreachable();
    }

    TriggerMode mode_;
    unsigned int multiplier_;
    Number interval_;
    std::vector<Number> time_points_;

    bool fired_ = false;
    unsigned int last_cycle_ = 0;
    Number last_t_ = Number(0.);
    Number last_wall_time_ = Number(0.);
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
     *  - prepare() after start() and whenever a task of the adapt phase
     *    sets TaskContext::prepare_required,
     *  - execute() in the phase of the task in every cycle,
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
       * Initialize the task.
       *
       * Called once at the very beginning.
       */
      virtual void initialize(TaskContext<Number> & /*context*/) {}

      /**
       * Start the computation.
       *
       * Called once after all tasks have been initialized and before the
       * first call to prepare().
       */
      virtual void start(TaskContext<Number> & /*context*/) {}

      /**
       * Prepare the task for the current mesh.
       *
       * Called after start() and whenever a task of the adapt phase sets
       * TaskContext::prepare_required.
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
       * Execute the task. Called in every cycle in the selected phase of
       * the task.
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
       * all MPI ranks; but only the data of rank 0 is written.
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

      /**
       * Return whether the task has a Trigger.
       */
      bool has_trigger() const
      {
        return trigger_ != nullptr;
      }

      /**
       * The Trigger that decides when the task performs its work. Must
       * only be called if has_trigger() returns true.
       */
      ACCESSOR_READ_ONLY(trigger)

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

      /**
       * The Trigger of the task. A derived class that needs one sets it
       * in its constructor.
       */
      std::unique_ptr<Trigger<Number>> trigger_;

    private:
      std::string name_;
      TaskPhase phase_;
    };
  } // namespace Interface
} // namespace ryujin
