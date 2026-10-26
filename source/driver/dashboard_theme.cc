//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <driver/dashboard_theme.h>

namespace ryujin::dashboard
{
  using namespace ftxui;

  namespace
  {
    Color hex(std::uint32_t rgb)
    {
      return Color::RGB((rgb >> 16) & 0xff, (rgb >> 8) & 0xff, rgb & 0xff);
    }
  } // namespace


  Palette make_palette(DashboardColorScheme scheme)
  {
    switch (scheme) {
    case DashboardColorScheme::ryujin:
      /* Shades of blue, and yellow, orange and red for the skew: */
      return {
          .badge = hex(0x7aa2f7),
          .badge_ink = hex(0x1a1b26),
          .ink = hex(0x1a1b26),
          .accent = hex(0x7aa2f7),
          .highlight = hex(0x2ac3de),
          .caption = hex(0x7dcfff),
          .good = hex(0xb4f9f8),
          .warn = hex(0x89ddff),
          .bad = hex(0xc0caf5),
          .muted = hex(0x565f89),
          .track = hex(0x292e42),
          .segments = {hex(0x7aa2f7),
                       hex(0x2ac3de),
                       hex(0x3d59a1),
                       hex(0x7dcfff),
                       hex(0x0db9d7),
                       hex(0x89ddff),
                       hex(0x394b70),
                       hex(0xb4f9f8),
                       hex(0xa9b1d6),
                       hex(0xc0caf5)},
          .animation = {hex(0x7dcfff),
                        hex(0x7aa2f7),
                        hex(0x89ddff),
                        hex(0xb4f9f8),
                        hex(0xc0caf5)},
          .heat = {hex(0x7aa2f7), hex(0xe0af68), hex(0xff9e64), hex(0xf7768e)},
      };

    case DashboardColorScheme::rainbow:
      /* Saturated colors of the full spectrum: */
      return {
          .badge = hex(0xd500f9),
          .badge_ink = hex(0x1a1b26),
          .ink = hex(0x1a1b26),
          .accent = hex(0x00e5ff),
          .highlight = hex(0xffea00),
          .caption = hex(0xb388ff),
          .good = hex(0x00e676),
          .warn = hex(0xff9100),
          .bad = hex(0xff1744),
          .muted = hex(0x6c6f9a),
          .track = hex(0x292e42),
          .segments = {hex(0xff1744),
                       hex(0xff9100),
                       hex(0xffea00),
                       hex(0x76ff03),
                       hex(0x00e676),
                       hex(0x00e5ff),
                       hex(0x2979ff),
                       hex(0x651fff),
                       hex(0xd500f9),
                       hex(0xff4081)},
          .animation = {hex(0xff1744),
                        hex(0xff9100),
                        hex(0xffea00),
                        hex(0x00e676),
                        hex(0x00e5ff),
                        hex(0x2979ff),
                        hex(0xd500f9)},
          .heat = {hex(0x00e676), hex(0xffea00), hex(0xff9100), hex(0xff1744)},
      };

    case DashboardColorScheme::tamu:
      /*
       * Aggie Maroon (#500000), white, and the neutrals of the Texas
       * A&M brand. Text uses tints of Aggie Maroon that are readable on
       * a dark terminal:
       */
      return {
          .badge = hex(0x500000),
          .badge_ink = hex(0xffffff),
          .ink = hex(0x202020),
          .accent = hex(0x966666),
          .highlight = hex(0xc2a6a6),
          .caption = hex(0xdccccc),
          .good = hex(0x84ee53),
          .warn = hex(0xeebd53),
          .bad = hex(0xee5389),
          .muted = hex(0x707070),
          .track = hex(0x3e3e3e),
          .segments = {hex(0x500000),
                       hex(0x966666),
                       hex(0x707070),
                       hex(0xc2a6a6),
                       hex(0xb0b0b0),
                       hex(0x733232),
                       hex(0xdccccc),
                       hex(0x4a4a4a),
                       hex(0xa87979),
                       hex(0xd9d9d9)},
          .animation = {hex(0x966666),
                        hex(0xc2a6a6),
                        hex(0xdccccc),
                        hex(0xffffff)},
          .heat = {hex(0x84ee53), hex(0xeebd53), hex(0xee5389)},
      };

    case DashboardColorScheme::monochrome:
      return {
          .badge = hex(0xd0d0d0),
          .badge_ink = hex(0x1c1c1c),
          .ink = hex(0x1c1c1c),
          .accent = hex(0xbcbcbc),
          .highlight = hex(0x9e9e9e),
          .caption = hex(0xd0d0d0),
          .good = hex(0xe4e4e4),
          .warn = hex(0xb2b2b2),
          .bad = hex(0xffffff),
          .muted = hex(0x6c6c6c),
          .track = hex(0x303030),
          .segments = {hex(0xeeeeee),
                       hex(0xbcbcbc),
                       hex(0x8a8a8a),
                       hex(0x5f5f5f),
                       hex(0xd0d0d0),
                       hex(0xa8a8a8),
                       hex(0x767676),
                       hex(0xe4e4e4),
                       hex(0x949494),
                       hex(0x4e4e4e)},
          .animation = {hex(0x6c6c6c),
                        hex(0x9e9e9e),
                        hex(0xd0d0d0),
                        hex(0xffffff)},
          .heat = {hex(0xe4e4e4), hex(0xb2b2b2), hex(0xffffff)},
      };
    }

    __builtin_unreachable();
  }


  Palette theme = make_palette(DashboardColorScheme::ryujin);


  Element badge()
  {
    return text("  ryujin  ") | bold | color(theme.badge_ink) |
           bgcolor(theme.badge);
  }


  Element panel(const std::string &title, Element content)
  {
    return window(text(" " + title + " ") | bold | color(theme.accent),
                  content | color(Color::Default)) |
           color(theme.muted);
  }


  Element label(const std::string &name)
  {
    return text(name) | color(theme.caption) | size(WIDTH, EQUAL, 11);
  }


  Element chip(const std::string &name, Color foreground)
  {
    return text(" " + name + " ") | color(foreground) | bgcolor(theme.track);
  }
} // namespace ryujin::dashboard
