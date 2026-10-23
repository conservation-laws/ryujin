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
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ryujin
{
  /**
   * Whether the Logger draws the Dashboard.
   *
   * @ingroup Driver
   */
  enum class DashboardMode : std::uint8_t {
    /**
     * Draw the dashboard if stdout is a terminal.
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
     * Fill the space around the dashboard with sparkles that change with
     * every update.
     */
    bool sparkles = true;
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
    double cpu_time_skew = 0.;
    double cpu_time_skew_fraction = 0.;

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
     * Destructor. Stops the background thread.
     */
    ~Dashboard();

    /**
     * Draw the snapshot @p data on the terminal with the dimensions,
     * placement and colors given by @p settings, and record it in the
     * throughput history.
     *
     * The first call starts a background thread that redraws the last
     * snapshot as soon as the terminal is resized, and that lets the
     * sparkles twinkle in between two calls. The thread stops with the
     * final snapshot (DashboardData::final_time).
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
     * Return a string of ANSI escape sequences that replaces some of the
     * sparkles of the last frame with new ones.
     */
    std::string twinkle();

    /**
     * The main loop of the background thread.
     */
    void animate();

    /**
     * Stop the background thread.
     */
    void stop();

    int terminal_width_ = 0;
    int terminal_height_ = 0;

    std::mt19937 random_engine_;

    std::deque<double> wall_history_;
    std::deque<double> compute_history_;

    /* The last snapshot, redrawn by the background thread on resize: */
    DashboardData data_;
    DashboardSettings settings_;

    /* The placement of the last frame and the positions of its sparkles: */
    int columns_ = 0;
    int rows_ = 0;
    int left_ = 0;
    int top_ = 0;
    int width_ = 0;
    int height_ = 0;
    std::vector<std::pair<int, int>> sparkles_;

    /* Guards all of the above, and the output to the terminal: */
    std::mutex mutex_;
    std::condition_variable condition_;
    std::thread thread_;
    bool stop_ = false;
  };
} // namespace ryujin
