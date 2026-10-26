//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/driver/dashboard.h>

#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ryujin::dashboard
{
  /**
   * The colors of a DashboardColorScheme.
   *
   * @ingroup Driver
   */
  struct Palette {
    /** The background of the header badge. */
    ftxui::Color badge;
    /** The text of the header badge. */
    ftxui::Color badge_ink;
    /** Text on light backgrounds. */
    ftxui::Color ink;
    /** Panel titles and the header border. */
    ftxui::Color accent;
    /** Chips, bars, and the "running" badge. */
    ftxui::Color highlight;
    /** Labels and captions. */
    ftxui::Color caption;
    /** Good values and states. */
    ftxui::Color good;
    /** Values and states that need attention. */
    ftxui::Color warn;
    /** Bad values, errors, and invalid entries. */
    ftxui::Color bad;
    /** Secondary text and panel borders. */
    ftxui::Color muted;
    /** The background of bars and gauges. */
    ftxui::Color track;
    /** The colors of the timers in the composition bar. */
    std::vector<ftxui::Color> segments;
    /** The colors of the animation around the dashboard. */
    std::vector<ftxui::Color> animation;
    /** The color scale of the skew, from none to large. */
    std::vector<ftxui::Color> heat;
  };


  /** Return the palette of the color scheme @p scheme. */
  Palette make_palette(DashboardColorScheme scheme);


  /** The palette of the elements below, set before rendering. */
  extern Palette theme;


  /** The "ryujin" badge of the header. */
  ftxui::Element badge();


  /** A panel with a title. */
  ftxui::Element panel(const std::string &title, ftxui::Element content);


  /** A caption of fixed width in front of a value. */
  ftxui::Element label(const std::string &name);


  /** A short text on the track color. */
  ftxui::Element chip(const std::string &name, ftxui::Color foreground);
} // namespace ryujin::dashboard
