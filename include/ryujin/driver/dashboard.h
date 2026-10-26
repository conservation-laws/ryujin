//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/patterns_conversion.h>

#include <deal.II/base/mpi.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ryujin::animation
{
  class Animation;
  struct Stage;
} // namespace ryujin::animation

namespace ryujin
{
  /**
   * Options controlling whether the Logger draws the Dashboard.
   *
   * @ingroup Driver
   */
  enum class DashboardMode : std::uint8_t {
    /**
     * Draw the dashboard if stdout reaches the terminal, directly or
     * through mpirun.
     */
    automatic,

    /**
     * Always draw the dashboard.
     */
    on,

    /**
     * Never draw the dashboard, print plain text instead.
     */
    off,
  };


  /**
   * The location of the Dashboard in the terminal.
   *
   * @ingroup Driver
   */
  enum class DashboardLocation : std::uint8_t {
    /**
     * The top left corner of the terminal.
     */
    top_left,

    /**
     * The center of the terminal.
     */
    center,

    /**
     * A random position that changes every ten seconds.
     */
    screen_saver,
  };


  /**
   * The color scheme of the Dashboard.
   *
   * @ingroup Driver
   */
  enum class DashboardColorScheme : std::uint8_t {
    /**
     * The default colors of ryujin: shades of blue, with yellow, orange
     * and red only for the skew of the timers.
     */
    ryujin,

    /**
     * Saturated colors of the full spectrum.
     */
    rainbow,

    /**
     * Aggie Maroon and the neutrals of the Texas A&M University brand.
     */
    tamu,

    /**
     * Shades of gray.
     */
    monochrome,
  };


  /**
   * The animation in the space around the Dashboard.
   *
   * @ingroup Driver
   */
  enum class DashboardAnimation : std::uint8_t {
    /**
     * No animation.
     */
    none,

    /**
     * Sparkles that twinkle in between two updates.
     */
    sparkle,

    /**
     * Snakes that roam around the dashboard and hunt for food.
     */
    matthias,
  };


  /**
   * The display of the wall throughput on the Dashboard.
   *
   * @ingroup Driver
   */
  enum class DashboardThroughputDisplay : std::uint8_t {
    /**
     * A speedometer with a needle.
     */
    speedometer,

    /**
     * A plain seven-segment display.
     */
    digital,
  };
} // namespace ryujin


#ifndef DOXYGEN
DECLARE_ENUM(ryujin::DashboardMode,
             LIST({ryujin::DashboardMode::automatic, "auto"},
                  {ryujin::DashboardMode::on, "on"},
                  {ryujin::DashboardMode::off, "off"}, ));

DECLARE_ENUM(ryujin::DashboardLocation,
             LIST({ryujin::DashboardLocation::top_left, "top left"},
                  {ryujin::DashboardLocation::center, "center"},
                  {ryujin::DashboardLocation::screen_saver, "screen saver"}, ));

DECLARE_ENUM(ryujin::DashboardColorScheme,
             LIST({ryujin::DashboardColorScheme::ryujin, "ryujin"},
                  {ryujin::DashboardColorScheme::rainbow, "rainbow"},
                  {ryujin::DashboardColorScheme::tamu, "tamu"},
                  {ryujin::DashboardColorScheme::monochrome, "monochrome"}, ));

DECLARE_ENUM(ryujin::DashboardAnimation,
             LIST({ryujin::DashboardAnimation::none, "none"},
                  {ryujin::DashboardAnimation::sparkle, "sparkle"},
                  {ryujin::DashboardAnimation::matthias, "matthias"}, ));

DECLARE_ENUM(ryujin::DashboardThroughputDisplay,
             LIST({ryujin::DashboardThroughputDisplay::speedometer,
                   "speedometer"},
                  {ryujin::DashboardThroughputDisplay::digital, "digital"}, ));
#endif

namespace ryujin
{
  /**
   * Run time settings of the Dashboard.
   *
   * @ingroup Driver
   */
  struct DashboardSettings {
    /**
     * Whether to draw the dashboard.
     */
    DashboardMode mode = DashboardMode::automatic;

    /**
     * The maximal width of the dashboard in columns. The dashboard fills
     * the terminal up to this width. Zero removes the limit.
     */
    unsigned int width = 140;

    /**
     * The maximal height of the dashboard in lines. If nonzero, the
     * dashboard stretches to fill the terminal up to this height. Zero
     * keeps the natural height of the dashboard.
     */
    unsigned int height = 0;

    /**
     * The location of the dashboard in the terminal.
     */
    DashboardLocation location = DashboardLocation::center;

    /**
     * The color scheme.
     */
    DashboardColorScheme color_scheme = DashboardColorScheme::ryujin;

    /**
     * The animation in the space around the dashboard.
     */
    DashboardAnimation animation = DashboardAnimation::sparkle;

    /**
     * The display of the wall throughput.
     */
    DashboardThroughputDisplay throughput_display =
        DashboardThroughputDisplay::digital;
  };


  /**
   * The position of the Dashboard, or of the parameter editor, in the
   * terminal. For DashboardLocation::screen_saver the position changes
   * to a random one every ten seconds.
   *
   * @ingroup Driver
   */
  class DashboardPosition
  {
  public:
    /**
     * Return the offset (left, top) of a window at @p location, with
     * @p free_columns and @p free_rows of free space around it.
     */
    std::pair<int, int>
    offset(DashboardLocation location, int free_columns, int free_rows) const;

    /**
     * For DashboardLocation::screen_saver, change to a new random
     * position if the last change, or the last call to postpone(), is
     * ten seconds ago. Return whether the position has changed.
     */
    bool move(DashboardLocation location);

    /**
     * Postpone the next change of the position by ten seconds.
     */
    void postpone();

  private:
    /** The fractions of the free space to the left and above. */
    std::pair<double, double> fractions_ = {0.5, 0.5};
    /** When the position changes next. */
    std::chrono::steady_clock::time_point next_move_;
  };


  /**
   * A snapshot of the cycle statistics gathered by the Logger on rank 0.
   * We either render it with the Dashboard or create a plain text info
   * from it.
   *
   * @ingroup Driver
   */
  struct DashboardData {
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
    /** The corrections of the hyperbolic and the parabolic module. */
    std::array<unsigned int, 2> corrections = {};
    /** The statistics of the linear solvers of the parabolic module. */
    std::vector<std::string> solver_statistics;
    /** The average time step size since the last snapshot. */
    double tau = 0.;
    /** The simulation time advanced per second of wall time. */
    double time_per_second = 0.;

    //@}
  };


  /**
   * A terminal dashboard for the cycle statistics recorded in a
   * DashboardData snapshot. The dashboard keeps a short history of
   * the throughput between calls to render().
   *
   * @ingroup Driver
   */
  class Dashboard
  {
  public:
    /**
     * Constructor.
     */
    Dashboard();

    /**
     * Destructor. Stops the background thread.
     */
    ~Dashboard();

    /**
     * Start a background thread that plays the animation, on the whole
     * terminal until the first call to show(). Call it before show().
     */
    void start(const DashboardSettings &settings);

    /**
     * Draw the snapshot @p data on the terminal with the dimensions,
     * placement and colors given by @p settings, and record it in the
     * throughput history.
     *
     * The background thread redraws the last snapshot as soon as the
     * terminal is resized or the dashboard moves, and advances the
     * animation in between two calls. It stops with the final snapshot
     * (DashboardData::final_time).
     */
    void show(const DashboardData &data, const DashboardSettings &settings);

  private:
    /**
     * Render the snapshot @p data into a string of ANSI escape sequences
     * that redraws the whole terminal in place. The terminal size is
     * queried on every call; if it has changed the screen is cleared
     * first.
     */
    std::string render(const DashboardData &data,
                       const DashboardSettings &settings);

    /**
     * Render a frame without the dashboard, in which the animation covers
     * the whole terminal.
     */
    std::string render_animation();

    /**
     * Advance the animation of the last frame and return a string of ANSI
     * escape sequences that redraws the cells it has changed.
     */
    std::string step();

    /**
     * The main loop of the background thread.
     */
    void animate();

    /**
     * Stop the background thread.
     */
    void stop();

    /**
     * Return the number of columns and rows of the terminal.
     */
    std::pair<int, int> terminal_size() const;

    /**
     * Write @p output to the terminal.
     */
    void write(const std::string &output) const;

    /** The terminal width of the last frame. */
    int terminal_width_ = 0;
    /** The terminal height of the last frame. */
    int terminal_height_ = 0;

    /** The recent wall throughput, oldest first. */
    std::deque<double> wall_history_;
    /** The recent compute throughput, oldest first. */
    std::deque<double> compute_history_;

    /** The last snapshot, redrawn by the background thread on resize. */
    DashboardData data_;
    /** The settings of the last snapshot. */
    DashboardSettings settings_;

    /** The position of the dashboard in the terminal. */
    DashboardPosition position_;

    /** Whether there is no snapshot yet and the animation plays alone. */
    bool animation_only_ = false;

    /** The last frame. */
    std::unique_ptr<animation::Stage> stage_;
    /** The animation of the last frame. */
    std::unique_ptr<animation::Animation> animation_;

    /**
     * The terminal (/dev/tty), written to directly instead of through
     * stdout. Under mpirun, stdout of rank 0 is forwarded by mpirun,
     * which delays the output if all cores are busy. -1 if there is no
     * controlling terminal, in which case we write to stdout.
     */
    int terminal_ = -1;

    /** Guards all of the above, and the output to the terminal. */
    std::mutex mutex_;
    /** Wakes the background thread when it shall stop. */
    std::condition_variable condition_;
    /** The background thread that plays the animation. */
    std::thread thread_;
    /** Whether the background thread shall stop. */
    bool stop_ = false;
  };
} // namespace ryujin
