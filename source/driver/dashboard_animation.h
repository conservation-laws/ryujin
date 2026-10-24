//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/driver/dashboard.h>

#include <ftxui/screen/box.hpp>
#include <ftxui/screen/color.hpp>
#include <ftxui/screen/screen.hpp>

#include <memory>
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
    ftxui::Screen screen;
    /* The cells covered by the dashboard: */
    ftxui::Box dashboard;
    /* The animation colors of the color scheme: */
    std::vector<ftxui::Color> colors;

    bool free(int x, int y) const
    {
      return !dashboard.Contain(x, y);
    }
  };


  /**
   * An animation in the space around the Dashboard.
   *
   * @ingroup Driver
   */
  class Animation
  {
  public:
    virtual ~Animation() = default;

    /**
     * Fill the free cells of a freshly rendered stage.
     */
    virtual void draw(Stage &stage, const DashboardData &data) = 0;

    /**
     * Advance the animation in between two snapshots.
     */
    virtual void step(Stage &stage, const DashboardData &data) = 0;
  };


  /**
   * Return the animation @p kind, or a null pointer for
   * DashboardAnimation::none.
   */
  std::unique_ptr<Animation> create(DashboardAnimation kind);
} // namespace ryujin::animation
