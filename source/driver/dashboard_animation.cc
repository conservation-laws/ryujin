//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <driver/dashboard_animation.h>

#include <algorithm>
#include <random>
#include <string>
#include <utility>

namespace ryujin::animation
{
  namespace
  {
    /**
     * Sparkles scattered over the free cells that wander off to other
     * free cells.
     */
    class Sparkle final : public Animation
    {
    public:
      void draw(Stage &stage, const DashboardData & /*data*/) override
      {
        constexpr double density = 0.02;

        sparkles_.clear();
        for (int y = 0; y < stage.screen.dimy(); ++y)
          for (int x = 0; x < stage.screen.dimx(); ++x)
            if (stage.free(x, y) && chance(density)) {
              sparkles_.emplace_back(x, y);
              light(stage, x, y);
            }
      }

      void step(Stage &stage, const DashboardData & /*data*/) override
      {
        if (sparkles_.empty())
          return;

        /* Move about one in eight sparkles to a free cell: */
        const auto n = std::max<std::size_t>(1, sparkles_.size() / 8);
        for (std::size_t i = 0; i < n; ++i) {
          auto &[x, y] = sparkles_[pick(sparkles_.size())];
          stage.screen.PixelAt(x, y) = ftxui::Pixel();

          for (int attempt = 0; attempt < 16; ++attempt) {
            const int new_x = int(pick(stage.screen.dimx()));
            const int new_y = int(pick(stage.screen.dimy()));
            if (stage.free(new_x, new_y) &&
                std::ranges::find(sparkles_, std::pair{new_x, new_y}) ==
                    sparkles_.end()) {
              x = new_x;
              y = new_y;
              break;
            }
          }

          light(stage, x, y);
        }
      }

    private:
      void light(Stage &stage, int x, int y)
      {
        static const std::vector<std::string> glyphs = {
            "·", "·", "⋅", "+", "*", "⋆", "✦", "✧", "˚"};

        auto &pixel = stage.screen.PixelAt(x, y);
        pixel = ftxui::Pixel();
        pixel.character = glyphs[pick(glyphs.size())];
        pixel.foreground_color = stage.colors[pick(stage.colors.size())];
        pixel.dim = chance(0.5);
      }

      /* Return a random index in [0, n): */
      std::size_t pick(std::size_t n)
      {
        return std::uniform_int_distribution<std::size_t>(0, n - 1)(random_);
      }

      bool chance(double p)
      {
        return std::uniform_real_distribution<double>(0., 1.)(random_) < p;
      }

      std::mt19937 random_;
      std::vector<std::pair<int, int>> sparkles_;
    };
  } // namespace


  std::unique_ptr<Animation> create(DashboardAnimation kind)
  {
    switch (kind) {
    case DashboardAnimation::none:
      return nullptr;
    case DashboardAnimation::sparkle:
      return std::make_unique<Sparkle>();
    }

    __builtin_unreachable();
  }
} // namespace ryujin::animation
