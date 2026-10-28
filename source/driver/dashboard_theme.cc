//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <driver/dashboard_theme.h>
#include <ryujin/base/version_info.h>

#include <string>
#include <utility>

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

    case DashboardColorScheme::ansi16:
      /*
       * The 16 colors of the terminal. Note that interpolated colors
       * fall back to the default RGB values of the palette:
       */
      return {
          .badge = Color::Blue,
          .badge_ink = Color::Black,
          .ink = Color::Black,
          .accent = Color::BlueLight,
          .highlight = Color::Cyan,
          .caption = Color::CyanLight,
          .good = Color::Green,
          .warn = Color::Yellow,
          .bad = Color::Red,
          .muted = Color::GrayLight,
          .track = Color::GrayDark,
          .segments = {Color::Blue,
                       Color::Cyan,
                       Color::Green,
                       Color::Yellow,
                       Color::Magenta,
                       Color::Red,
                       Color::BlueLight,
                       Color::CyanLight,
                       Color::GreenLight,
                       Color::MagentaLight},
          .animation = {Color::BlueLight,
                        Color::CyanLight,
                        Color::MagentaLight,
                        Color::White},
          .heat = {Color::Blue, Color::Yellow, Color::RedLight, Color::Red},
      };

    case DashboardColorScheme::gruvbox:
      /*
       * The warm colors of the dark Gruvbox palette,
       * https://github.com/morhetz/gruvbox
       */
      return {
          .badge = hex(0xfabd2f),
          .badge_ink = hex(0x282828),
          .ink = hex(0x282828),
          .accent = hex(0xfe8019),
          .highlight = hex(0x8ec07c),
          .caption = hex(0xd5c4a1),
          .good = hex(0xb8bb26),
          .warn = hex(0xfabd2f),
          .bad = hex(0xfb4934),
          .muted = hex(0x928374),
          .track = hex(0x3c3836),
          .segments = {hex(0xfb4934),
                       hex(0xfe8019),
                       hex(0xfabd2f),
                       hex(0xb8bb26),
                       hex(0x8ec07c),
                       hex(0x83a598),
                       hex(0xd3869b),
                       hex(0xd65d0e),
                       hex(0x689d6a),
                       hex(0x458588)},
          .animation = {hex(0xfe8019),
                        hex(0xfabd2f),
                        hex(0xb8bb26),
                        hex(0x8ec07c),
                        hex(0x83a598)},
          .heat = {hex(0x83a598), hex(0xfabd2f), hex(0xfe8019), hex(0xfb4934)},
      };

    case DashboardColorScheme::light:
      /*
       * The ryujin color scheme with darker shades of blue for a light
       * background, and yellow, orange and red for the skew. Derived from
       * the Tokyo Night Day palette, https://github.com/folke/tokyonight.nvim
       */
      return {
          .badge = hex(0x2e7de9),
          .badge_ink = hex(0xffffff),
          .ink = hex(0xffffff),
          .accent = hex(0x2e5cb8),
          .highlight = hex(0x2e7de9),
          .caption = hex(0x34548a),
          .good = hex(0x007197),
          .warn = hex(0x006a83),
          .bad = hex(0x3760bf),
          .muted = hex(0x8990b3),
          .track = hex(0xc4c8da),
          .segments = {hex(0x2e7de9),
                       hex(0x007197),
                       hex(0x3760bf),
                       hex(0x188092),
                       hex(0x6172b0),
                       hex(0x2e5cb8),
                       hex(0x9854f1),
                       hex(0x0f4b6e),
                       hex(0x7890dd),
                       hex(0x34548a)},
          .animation = {hex(0x2e7de9),
                        hex(0x007197),
                        hex(0x188092),
                        hex(0x9854f1),
                        hex(0x3760bf)},
          .heat = {hex(0x2e7de9), hex(0x8c6c3e), hex(0xb15c00), hex(0xf52a65)},
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

    case DashboardColorScheme::okabe_ito:
      /*
       * The Okabe-Ito palette that stays distinguishable for color
       * vision deficiencies, https://jfly.uni-koeln.de/color/, extended
       * by tints of sky blue and orange:
       */
      return {
          .badge = hex(0x56b4e9),
          .badge_ink = hex(0x1a1b26),
          .ink = hex(0x1a1b26),
          .accent = hex(0x56b4e9),
          .highlight = hex(0xe69f00),
          .caption = hex(0xc8c8c8),
          .good = hex(0x009e73),
          .warn = hex(0xf0e442),
          .bad = hex(0xd55e00),
          .muted = hex(0x6c6c6c),
          .track = hex(0x303030),
          .segments = {hex(0xe69f00),
                       hex(0x56b4e9),
                       hex(0x009e73),
                       hex(0xf0e442),
                       hex(0x0072b2),
                       hex(0xd55e00),
                       hex(0xcc79a7),
                       hex(0x999999),
                       hex(0xa6d6f2),
                       hex(0xf2c266)},
          .animation = {hex(0xe69f00),
                        hex(0x56b4e9),
                        hex(0x009e73),
                        hex(0xf0e442),
                        hex(0xcc79a7)},
          .heat = {hex(0x56b4e9), hex(0xf0e442), hex(0xe69f00), hex(0xd55e00)},
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

    case DashboardColorScheme::solarized:
      /*
       * The accent colors of the Solarized palette, with the background
       * highlight of the dark variant for bars and gauges,
       * https://ethanschoonover.com/solarized/
       */
      return {
          .badge = hex(0x268bd2),
          .badge_ink = hex(0xfdf6e3),
          .ink = hex(0x002b36),
          .accent = hex(0x268bd2),
          .highlight = hex(0x2aa198),
          .caption = hex(0x93a1a1),
          .good = hex(0x859900),
          .warn = hex(0xb58900),
          .bad = hex(0xdc322f),
          .muted = hex(0x586e75),
          .track = hex(0x073642),
          .segments = {hex(0x268bd2),
                       hex(0x2aa198),
                       hex(0x859900),
                       hex(0xb58900),
                       hex(0xcb4b16),
                       hex(0xdc322f),
                       hex(0xd33682),
                       hex(0x6c71c4),
                       hex(0x93a1a1),
                       hex(0x839496)},
          .animation = {hex(0x268bd2),
                        hex(0x2aa198),
                        hex(0x859900),
                        hex(0xb58900),
                        hex(0xcb4b16),
                        hex(0xd33682),
                        hex(0x6c71c4)},
          .heat = {hex(0x268bd2), hex(0xb58900), hex(0xcb4b16), hex(0xdc322f)},
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
    }

    __builtin_unreachable();
  }


  Palette theme = make_palette(DashboardColorScheme::ryujin);


  Element badge(bool with_version)
  {
    const std::string name = with_version ? version_string() : "ryujin";
    return text("  " + name + "  ") | bold | color(theme.badge_ink) |
           bgcolor(theme.badge);
  }


  Element panel(const std::string &title, Element content)
  {
    return window(text(" " + title + " ") | bold | color(theme.accent),
                  std::move(content) | color(Color::Default)) |
           color(theme.muted);
  }


  Element caption(const std::string &text)
  {
    return ftxui::text(text) | color(theme.caption);
  }


  Element muted(const std::string &text)
  {
    return ftxui::text(text) | color(theme.muted);
  }


  Element label(const std::string &name)
  {
    return caption(name) | size(WIDTH, EQUAL, 11);
  }


  Element chip(const std::string &name, Color foreground)
  {
    return text(" " + name + " ") | color(foreground) | bgcolor(theme.track);
  }


  Element place(Element element, int left, int top)
  {
    return vbox({
        emptyElement() | size(HEIGHT, EQUAL, top),
        hbox({emptyElement() | size(WIDTH, EQUAL, left), std::move(element)}),
    });
  }
} // namespace ryujin::dashboard
