//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <driver/dashboard_frame.h>
#include <driver/dashboard_music.h>
#include <driver/dashboard_theme.h>
#include <ryujin/base/random.h>

#include <algorithm>
#include <random>

namespace ryujin::dashboard
{
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


  void Frame::draw(animation::Stage &stage)
  {
    if (animation_)
      animation_->draw(stage);
  }


  bool Frame::step(animation::Stage &stage)
  {
    if (!animation_)
      return false;

    stage.changed.clear();
    animation_->step(stage);
    if (stage.shift == std::pair{0, 0})
      return stage.redraw;

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
