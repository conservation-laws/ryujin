//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <dashboard/animation.h>
#include <dashboard/settings.h>

#include <chrono>
#include <memory>
#include <utility>

namespace ryujin::dashboard
{
  /**
   * The frame of a window on the terminal, shared by the Dashboard and the
   * parameter editor: the DashboardSettings, the position of the window,
   * and the animation around it. For DashboardLocation::screen_saver the
   * position changes to a random one every ten seconds.
   *
   * @ingroup Driver
   */
  class Frame
  {
  public:
    /**
     * Take over the colors, the animation, and the music of @p settings.
     * The music only changes with the setting, so that the music of
     * SplashScreen::arcade keeps playing for DashboardMusic::none.
     */
    void configure(const DashboardSettings &settings);

    /**
     * The settings.
     */
    const DashboardSettings &settings() const
    {
      return settings_;
    }

    /**
     * Return the width of the window on a terminal with @p columns.
     */
    int width(int columns) const;

    /**
     * Return the offset (left, top) of the window with @p free_columns and
     * @p free_rows of free space around it.
     */
    std::pair<int, int> offset(int free_columns, int free_rows) const;

    /**
     * For DashboardLocation::screen_saver, change to a new random
     * position if the last change, or the last call to postpone(), is
     * ten seconds ago. Return whether the position has changed.
     */
    bool move();

    /**
     * Postpone the next change of the position by ten seconds.
     */
    void postpone();

    /**
     * Fill the free cells of the freshly rendered @p stage, and move the
     * window as the animation asks for. Return whether the whole stage
     * has to be rendered again.
     */
    bool draw(animation::Stage &stage);

    /**
     * Fill the free cells of the freshly rendered final @p stage, after
     * the simulation finished, and move the window as the animation asks
     * for. Return whether the whole stage has to be rendered again.
     */
    bool finish(animation::Stage &stage);

    /**
     * Advance the animation on @p stage, and move the window as the
     * animation asks for. Return whether the whole stage has to be
     * rendered again.
     */
    bool step(animation::Stage &stage);

    /**
     * Return whether there is an animation.
     */
    bool animated() const
    {
      return animation_ != nullptr;
    }

  private:
    /**
     * Move the window by the shift that the animation asked for on
     * @p stage. Return whether the window has moved.
     */
    bool shift(const animation::Stage &stage);

    /** The settings. */
    DashboardSettings settings_;

    /** The animation around the window, null for none. */
    std::unique_ptr<animation::Animation> animation_;

    /** The fractions of the free space to the left and above. */
    std::pair<double, double> fractions_ = {0.5, 0.5};

    /** Whether an animation has moved the window. */
    bool shifted_ = false;

    /** When the screen saver moves next. */
    std::chrono::steady_clock::time_point next_move_;
  };
} // namespace ryujin::dashboard
