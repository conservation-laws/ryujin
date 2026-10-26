//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/driver/dashboard.h>

#include <ftxui/screen/box.hpp>
#include <ftxui/screen/screen.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace ryujin::animation
{
  /**
   * The stage of an animation: the whole terminal, of which the animation
   * may only change the cells outside of the dashboard.
   *
   * @ingroup Driver
   */
  struct Stage {
    /** The cells of the whole terminal. */
    ftxui::Screen screen;
    /** The cells covered by the dashboard. */
    ftxui::Box dashboard;
    /** The cells changed by the animation since the last step. */
    std::vector<std::pair<int, int>> changed = {};

    /** Return whether the cell (x, y) is outside of the dashboard. */
    bool free(int x, int y) const
    {
      return !dashboard.Contain(x, y);
    }

    /** Clear the pixel at (x, y), mark it as changed, and return it. */
    ftxui::Pixel &pixel(int x, int y)
    {
      changed.emplace_back(x, y);
      return screen.PixelAt(x, y) = ftxui::Pixel();
    }
  };


  /**
   * An animation in the space around the Dashboard, in the animation
   * colors of dashboard::theme.
   *
   * @ingroup Driver
   */
  class Animation
  {
  public:
    virtual ~Animation() = default;

    /** Fill the free cells of a freshly rendered stage. */
    virtual void draw(Stage &stage) = 0;

    /**
     * Advance the animation. All changes to the stage have to go through
     * Stage::pixel().
     */
    virtual void step(Stage &stage) = 0;
  };


  /** Return the animation @p kind, null for DashboardAnimation::none. */
  std::unique_ptr<Animation> create(DashboardAnimation kind);
} // namespace ryujin::animation
