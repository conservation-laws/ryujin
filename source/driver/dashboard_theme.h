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
    /* header badge and its text: */
    ftxui::Color badge;
    ftxui::Color badge_ink;
    /* text on light backgrounds: */
    ftxui::Color ink;
    /* panel titles and the header border: */
    ftxui::Color accent;
    /* chips, bars, and the "running" badge: */
    ftxui::Color highlight;
    /* labels and captions: */
    ftxui::Color caption;
    ftxui::Color good;
    ftxui::Color warn;
    ftxui::Color bad;
    /* secondary text and panel borders: */
    ftxui::Color muted;
    /* the background of bars and gauges: */
    ftxui::Color track;
    std::vector<ftxui::Color> segments;
    std::vector<ftxui::Color> animation;
    /* the color scale of the skew, from none to large: */
    std::vector<ftxui::Color> heat;
  };


  /**
   * Return the palette of the color scheme @p scheme.
   */
  Palette make_palette(DashboardColorScheme scheme);


  /**
   * The palette used by the elements below, set by the Dashboard and the
   * parameter editor before rendering.
   */
  extern Palette theme;


  /**
   * The "ryujin" badge of the header.
   */
  ftxui::Element badge();


  /**
   * A panel with a title.
   */
  ftxui::Element panel(const std::string &title, ftxui::Element content);


  /**
   * A caption of fixed width in front of a value.
   */
  ftxui::Element label(const std::string &name);


  /**
   * A short text on the track color.
   */
  ftxui::Element chip(const std::string &name, ftxui::Color foreground);
} // namespace ryujin::dashboard
