//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/patterns_conversion.h>

#include <deal.II/base/mpi.h>

#include <array>
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
   * Whether the Logger draws the Dashboard.
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
     * A colorful variant of the ryujin colors.
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
     * Center the dashboard in the terminal.
     */
    bool center = true;

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
   * A snapshot of the cycle statistics gathered by the Logger on rank 0.
   * The Dashboard renders it to the terminal.
   *
   * @ingroup Driver
   */
  struct DashboardData {
    using MinMaxAvg = dealii::Utilities::MPI::MinMaxAvg;

    /**
     * Statistics of a single ComputingTimer.
     */
    struct Timer {
      std::string name;
      MinMaxAvg wall_time = {};
      double cpu_time = 0.;
      /* fraction of the total cpu time, negative if not meaningful: */
      double fraction = -1.;
    };

    /**
     * Status of an active task and its trigger.
     */
    struct Task {
      std::string name;
      std::string phase;
      /* the condition of the trigger, empty if the task has none: */
      std::string trigger;
      std::string last;
      std::string next;
      /* progress towards the next firing, negative if not meaningful: */
      double progress = -1.;
    };

    /**
     * A status message and the cycle and time at which it was issued.
     */
    struct Message {
      std::string text;
      unsigned int cycle = 0;
      double t = 0.;
    };

    /**
     * @name Header
     */
    //@{

    bool final_time = false;
    unsigned int cycle = 0;
    double t = 0.;
    double t_final = 0.;
    double eta = 0.;
    double terminal_update_interval = 0.;

    //@}
    /**
     * @name Information
     */
    //@{

    std::string hyperbolic_name;
    std::string parabolic_name;
    std::string base_name;
    std::string backend;
    std::string threads;
    double n_global_dofs = 0.;
    unsigned int n_ensembles = 1;
    unsigned int n_ranks = 1;
    unsigned int last_output_cycle = 0;
    double last_output_time = 0.;

    //@}
    /**
     * @name Messages
     */
    //@{

    std::deque<Message> messages;

    //@}
    /**
     * @name Memory, timers, and tasks
     */
    //@{

    MinMaxAvg memory = {};
    std::vector<Timer> timers;
    std::vector<Task> tasks;

    //@}
    /**
     * @name Throughput
     */
    //@{

    bool device = false;
    bool fudged_cpu_timings = false;
    unsigned int n_devices = 0;
    double compute_m_dofs_per_sec = 0.;
    double wall_m_dofs_per_sec = 0.;
    double cycles_per_second = 0.;
    double utilization = 0.;
    /* The device and wall time since the last snapshot: */
    double delta_device_time = 0.;
    double delta_wall_time = 0.;
    double cpu_time_skew = 0.;
    double cpu_time_skew_fraction = 0.;
    /* The skew of the device time over all ranks, and the ranks with the
     * smallest and largest accumulated device time: */
    double device_time_skew = 0.;
    double device_time_skew_fraction = 0.;
    unsigned int device_time_min_index = 0;
    unsigned int device_time_max_index = 0;

    std::string time_stepping_scheme;
    double cfl = 0.;
    /* Event counters of the hyperbolic and the parabolic module: */
    std::array<unsigned int, 2> restarts = {};
    std::array<unsigned int, 2> warnings = {};
    std::array<unsigned int, 2> corrections = {};
    std::vector<std::string> solver_statistics;
    double tau = 0.;
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
     * terminal is resized, and advances the animation in between two
     * calls. It stops with the final snapshot (DashboardData::final_time).
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

    int terminal_width_ = 0;
    int terminal_height_ = 0;

    std::deque<double> wall_history_;
    std::deque<double> compute_history_;

    /* The last snapshot, redrawn by the background thread on resize: */
    DashboardData data_;
    DashboardSettings settings_;
    /* Whether there is no snapshot yet and the animation plays alone: */
    bool animation_only_ = false;

    /* The last frame and its animation: */
    std::unique_ptr<animation::Stage> stage_;
    std::unique_ptr<animation::Animation> animation_;

    /*
     * The terminal (/dev/tty), written to directly instead of through
     * stdout. Under mpirun, stdout of rank 0 is forwarded by mpirun,
     * which delays the output if all cores are busy. -1 if there is no
     * controlling terminal, in which case we write to stdout:
     */
    int terminal_ = -1;

    /* Guards all of the above, and the output to the terminal: */
    std::mutex mutex_;
    std::condition_variable condition_;
    std::thread thread_;
    bool stop_ = false;
  };
} // namespace ryujin
