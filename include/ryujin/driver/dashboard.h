//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/patterns_conversion.h>

#include <deal.II/base/mpi.h>

#include <array>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

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
     * The 16 colors of the terminal, so that the colors follow the
     * theme of the terminal, and work without true color support.
     */
    ansi16,

    /**
     * The warm colors of the dark Gruvbox palette by Pavel Pertsev,
     * https://github.com/morhetz/gruvbox
     */
    gruvbox,

    /**
     * Shades of blue for terminals with a light background, derived
     * from the Tokyo Night Day palette by Folke Lemaitre,
     * https://github.com/folke/tokyonight.nvim
     */
    light,

    /**
     * Shades of gray.
     */
    monochrome,

    /**
     * The palette of M. Okabe and K. Ito, "Color Universal Design
     * (CUD): How to make figures and presentations that are friendly to
     * colorblind people" (2008), https://jfly.uni-koeln.de/color/, which
     * stays distinguishable for color vision deficiencies.
     */
    okabe_ito,

    /**
     * Saturated colors of the full spectrum.
     */
    rainbow,

    /**
     * The accent colors of the Solarized palette by Ethan Schoonover,
     * https://ethanschoonover.com/solarized/
     */
    solarized,

    /**
     * Aggie Maroon and the neutrals of the Texas A&M University brand.
     */
    tamu,
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
    ryujin,

    /**
     * Snakes that roam around the dashboard and hunt for food.
     */
    matthias,

    /**
     * An aquarium around the dashboard.
     */
    wyatt,
  };


  /**
   * The music played while the Dashboard is drawn.
   *
   * @ingroup Driver
   */
  enum class DashboardMusic : std::uint8_t {
    /**
     * No music.
     */
    none,

    /**
     * An original tune in the style of the arcade racing games of the
     * eighties, also played by SplashScreen::arcade.
     */
    ryujin,

    /**
     * An original maze chase in the style of the arcade games of the
     * eighties, for the snakes of DashboardAnimation::matthias.
     */
    matthias,

    /**
     * An original underwater tune in the style of the arcade games of the
     * eighties, for the aquarium of DashboardAnimation::wyatt.
     */
    wyatt,
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
                  {ryujin::DashboardColorScheme::ansi16, "ansi16"},
                  {ryujin::DashboardColorScheme::gruvbox, "gruvbox"},
                  {ryujin::DashboardColorScheme::light, "light"},
                  {ryujin::DashboardColorScheme::monochrome, "monochrome"},
                  {ryujin::DashboardColorScheme::okabe_ito, "okabe-ito"},
                  {ryujin::DashboardColorScheme::rainbow, "rainbow"},
                  {ryujin::DashboardColorScheme::solarized, "solarized"},
                  {ryujin::DashboardColorScheme::tamu, "tamu"}, ));

DECLARE_ENUM(ryujin::DashboardAnimation,
             LIST({ryujin::DashboardAnimation::none, "none"},
                  {ryujin::DashboardAnimation::ryujin, "ryujin"},
                  {ryujin::DashboardAnimation::matthias, "matthias"},
                  {ryujin::DashboardAnimation::wyatt, "wyatt"}, ));

DECLARE_ENUM(ryujin::DashboardMusic,
             LIST({ryujin::DashboardMusic::none, "none"},
                  {ryujin::DashboardMusic::ryujin, "ryujin"},
                  {ryujin::DashboardMusic::matthias, "matthias"},
                  {ryujin::DashboardMusic::wyatt, "wyatt"}, ));
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
    DashboardAnimation animation = DashboardAnimation::none;

    /**
     * The music played while the dashboard is drawn. DashboardMusic::none
     * keeps the music of SplashScreen::arcade playing.
     */
    DashboardMusic music = DashboardMusic::none;
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
} // namespace ryujin
