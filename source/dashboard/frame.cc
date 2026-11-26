//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <dashboard/frame.h>
#include <dashboard/music.h>
#include <dashboard/theme.h>
#include <ryujin/base/random.h>

#include <algorithm>
#include <random>
#include <set>

namespace ryujin::dashboard
{
  namespace
  {
    /* Return whether the cell (x, y) is a blank on the default background: */
    bool blank(const animation::Stage &stage, int x, int y)
    {
      const auto &p = stage.screen.PixelAt(x, y);
      return (p.character == " " || p.character.empty()) &&
             p.background_color == ftxui::Color::Default;
    }


    /*
     * The empty cells of the dashboard, with a cell of margin around text
     * that keeps the words apart from the glass:
     */
    std::set<std::pair<int, int>> glass(const animation::Stage &stage)
    {
      std::set<std::pair<int, int>> cells;
      const auto &d = stage.dashboard;
      for (int y = d.y_min; y <= d.y_max; ++y)
        for (int x = d.x_min + 1; x < d.x_max; ++x)
          if (blank(stage, x - 1, y) && blank(stage, x, y) &&
              blank(stage, x + 1, y))
            cells.emplace(x, y);
      return cells;
    }


    /* Fade what the animation painted into the pixel @p p of the glass: */
    void fade(ftxui::Pixel &p)
    {
      constexpr float strength = 0.8f;
      if (p.foreground_color != ftxui::Color::Default)
        p.foreground_color = ftxui::Color::Interpolate(
            strength, p.foreground_color, theme.track);
      if (p.background_color != ftxui::Color::Default)
        p.background_color = ftxui::Color::Interpolate(
            strength, p.background_color, theme.track);
      p.bold = false;
    }
  } // namespace


  void Frame::configure(const DashboardSettings &settings)
  {
    const auto previous = std::exchange(settings_, settings);

    theme = make_palette(settings.color_scheme);
    if (!animation_ || settings.animation != previous.animation)
      animation_ = animation::create(settings.animation);
    if (settings.music != previous.music)
      play_dashboard_music(settings.music);
    if (settings.location != previous.location)
      shifted_ = false;
  }


  int Frame::width(int columns) const
  {
    if (settings_.width == 0)
      return columns;
    return std::min(columns, std::max<int>(settings_.width, 80));
  }


  std::pair<int, int> Frame::offset(int free_columns, int free_rows) const
  {
    const auto location = settings_.location;
    const auto [x, y] =
        shifted_                                  ? fractions_
        : location == DashboardLocation::top_left ? std::pair{0., 0.}
        : location == DashboardLocation::center   ? std::pair{.5, .5}
                                                  : fractions_;
    return {static_cast<int>(x * free_columns),
            static_cast<int>(y * free_rows)};
  }


  bool Frame::move()
  {
    if (settings_.location != DashboardLocation::screen_saver ||
        std::chrono::steady_clock::now() < next_move_)
      return false;

    static auto random = seeded_mersenne_twister();
    std::uniform_real_distribution<double> uniform;
    fractions_ = {uniform(random), uniform(random)};
    postpone();
    return true;
  }


  void Frame::postpone()
  {
    next_move_ = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  }


  bool Frame::draw(animation::Stage &stage)
  {
    return paint(stage, false);
  }


  bool Frame::finish(animation::Stage &stage)
  {
    return paint(stage, true);
  }


  bool Frame::step(animation::Stage &stage)
  {
    if (!animation_ || !backdrop_)
      return false;

    stage.changed.clear();
    backdrop_->changed.clear();
    animation_->step(*backdrop_);
    for (const auto &[x, y] : backdrop_->changed) {
      compose(stage, x, y);
      stage.changed.emplace_back(x, y);
    }
    return shift(*backdrop_) || backdrop_->redraw;
  }


  bool Frame::paint(animation::Stage &stage, bool finish)
  {
    if (!animation_)
      return false;

    const int columns = stage.screen.dimx();
    const int rows = stage.screen.dimy();
    backdrop_.emplace(ftxui::Screen(columns, rows),
                      settings_.glass ? ftxui::Box{0, -1, 0, -1}
                                      : stage.dashboard,
                      stage.progress,
                      stage.eta);
    glass_.clear();
    if (settings_.glass)
      glass_ = glass(stage);

    if (finish)
      animation_->finish(*backdrop_);
    else
      animation_->draw(*backdrop_);
    for (int y = 0; y < rows; ++y)
      for (int x = 0; x < columns; ++x)
        compose(stage, x, y);
    return shift(*backdrop_);
  }


  void Frame::compose(animation::Stage &stage, int x, int y) const
  {
    const bool inside = stage.dashboard.Contain(x, y);
    if (inside && !glass_.contains({x, y}))
      return;

    auto &p = stage.screen.PixelAt(x, y) = backdrop_->screen.PixelAt(x, y);
    if (inside)
      fade(p);
  }


  bool Frame::shift(const animation::Stage &stage)
  {
    if (stage.shift == std::pair{0, 0})
      return false;

    /* Keep the new relative position, rounded down to the shifted cell: */
    const auto [free_columns, free_rows] = stage.free_space();
    const auto fraction = [](int x, int free) {
      return free > 0 ? std::clamp((x + 0.5) / free, 0., 1.) : 0.5;
    };
    const auto [left, top] = offset(free_columns, free_rows);
    fractions_ = {fraction(left + stage.shift.first, free_columns),
                  fraction(top + stage.shift.second, free_rows)};
    shifted_ = true;
    return true;
  }
} // namespace ryujin::dashboard
