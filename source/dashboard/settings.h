//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/patterns_conversion.h>

#include <cstdint>

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
     * The soft greens of the dark Everforest palette by Sainnhe Park,
     * https://github.com/sainnhe/everforest
     */
    everforest,

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
     * The Frost and Aurora colors of the Nord palette by Arctic Ice
     * Studio, https://www.nordtheme.com
     */
    nord,

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
     * A barista brewing a pour-over coffee that is ready when the
     * simulation finishes.
     */
    eric,

    /**
     * An aquarium around the dashboard.
     */
    wyatt,

    /**
     * A forest at night around the dashboard, with animals that roam the
     * forest floor.
     */
    jordan,
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
     * A not very aquatic waltz.
     */
    wyatt,

    /**
     * An original folk song in the style of the arcade games of the
     * eighties, for the forest at night of DashboardAnimation::jordan.
     */
    jordan,
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
                  {ryujin::DashboardColorScheme::everforest, "everforest"},
                  {ryujin::DashboardColorScheme::gruvbox, "gruvbox"},
                  {ryujin::DashboardColorScheme::light, "light"},
                  {ryujin::DashboardColorScheme::monochrome, "monochrome"},
                  {ryujin::DashboardColorScheme::nord, "nord"},
                  {ryujin::DashboardColorScheme::okabe_ito, "okabe-ito"},
                  {ryujin::DashboardColorScheme::rainbow, "rainbow"},
                  {ryujin::DashboardColorScheme::solarized, "solarized"},
                  {ryujin::DashboardColorScheme::tamu, "tamu"}, ));

DECLARE_ENUM(ryujin::DashboardAnimation,
             LIST({ryujin::DashboardAnimation::none, "none"},
                  {ryujin::DashboardAnimation::ryujin, "ryujin"},
                  {ryujin::DashboardAnimation::matthias, "matthias"},
                  {ryujin::DashboardAnimation::eric, "eric"},
                  {ryujin::DashboardAnimation::jordan, "jordan"},
                  {ryujin::DashboardAnimation::wyatt, "wyatt"}, ));

DECLARE_ENUM(ryujin::DashboardMusic,
             LIST({ryujin::DashboardMusic::none, "none"},
                  {ryujin::DashboardMusic::ryujin, "ryujin"},
                  {ryujin::DashboardMusic::matthias, "matthias"},
                  {ryujin::DashboardMusic::jordan, "jordan"},
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

    /**
     * Draw the animation on the whole terminal, behind the dashboard, and
     * let it show faintly through the empty cells of the dashboard, as
     * through smoked glass.
     */
    bool glass = false;
  };
} // namespace ryujin
