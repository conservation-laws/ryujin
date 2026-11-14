//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <driver/dashboard_frame.h>
#include <ryujin/driver/dashboard.h>
#include <ryujin/interface/task.h>

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>

namespace ryujin::terminal
{
  class Keyboard;
  class Output;
} // namespace ryujin::terminal

namespace ryujin
{
  /**
   * A terminal dashboard for the cycle statistics recorded in a
   * DashboardData snapshot. The dashboard keeps a short history of
   * the throughput.
   *
   * @ingroup Driver
   */
  class Dashboard
  {
  public:
    Dashboard();

    /**
     * Destructor. Stops the background thread.
     */
    ~Dashboard();

    /**
     * Start a background thread that plays the animation, on the whole
     * terminal until the first call to show().
     *
     * If keys can be read from stdin, the dashboard shows a menu, and
     * the background thread folds out two panels from it: "run", which
     * issues a TaskRequest, and "options", in which all settings but
     * DashboardSettings::mode can be changed while the dashboard is shown.
     */
    void start(const DashboardSettings &settings);

    /**
     * Draw the snapshot @p data on the terminal and record it in the
     * throughput history.
     *
     * The background thread redraws the last snapshot as soon as the
     * terminal is resized or the dashboard moves, and advances the
     * animation in between two calls. It stops with the final snapshot
     * (DashboardData::final_time).
     */
    void show(const DashboardData &data);

    /**
     * Return the request issued in the run panel since the last call, or
     * TaskRequest::none. The function does not wait for the background
     * thread.
     */
    TaskRequest take_request();

  private:
    /**
     * Render the last snapshot, or only the animation if there is none
     * yet, into a string of ANSI escape sequences that redraws the whole
     * terminal in place.
     */
    std::string render();

    /**
     * Handle the keys @p keys read by the background thread: select a
     * button of the menu, open and close the run and the options panel,
     * issue a request in the former and change the settings in the
     * latter.
     */
    void handle(const std::string &keys);

    /**
     * Return the number of columns and rows of the terminal.
     */
    std::pair<int, int> terminal_size() const;

    /**
     * Write @p output to the terminal.
     */
    void write(const std::string &output) const;

    /** The settings, the position, and the animation. */
    dashboard::Frame frame_;
    /** The last snapshot, redrawn by the background thread. */
    std::optional<DashboardData> data_;

    /** The recent wall throughput, oldest first. */
    std::deque<double> wall_history_;
    /** The recent compute throughput, oldest first. */
    std::deque<double> compute_history_;

    /** The terminal size of the last frame. */
    std::pair<int, int> terminal_size_;
    /** The last frame. */
    std::unique_ptr<animation::Stage> stage_;

    /** The output to the terminal. */
    std::unique_ptr<terminal::Output> output_;
    /** The keyboard input, null if keys cannot be read from stdin. */
    std::unique_ptr<terminal::Keyboard> keyboard_;

    /** The button selected in the menu below the dashboard. */
    unsigned int button_ = 0;
    /** The row selected in the run panel, if it is open. */
    std::optional<unsigned int> run_;
    /** Whether the request selected in the run panel awaits confirmation. */
    bool confirm_ = false;
    /** The row selected in the options panel, if it is open. */
    std::optional<unsigned int> option_;

    /** Guards all of the above, and the output to the terminal. */
    std::mutex mutex_;
    /** The request issued in the run panel, taken by take_request(). */
    std::atomic<TaskRequest> request_ = TaskRequest::none;
    /** The background thread that plays the animation. */
    std::jthread thread_;
  };
} // namespace ryujin
