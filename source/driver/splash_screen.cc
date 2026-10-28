//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <driver/splash_screen_logo.h>
#include <driver/terminal.h>
#include <ryujin/base/version_info.h>
#include <ryujin/driver/splash_screen.h>

#include <ftxui/screen/color.hpp>
#include <ftxui/screen/screen.hpp>
#include <ftxui/screen/string.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <format>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace ryujin
{
  namespace
  {
    using ftxui::Color;
    using ftxui::Screen;

    /*
     * Colors:
     */

    struct RGB {
      float r = 0.f;
      float g = 0.f;
      float b = 0.f;
    };

    RGB mix(const RGB &a, const RGB &b, float t)
    {
      t = std::clamp(t, 0.f, 1.f);
      return {
          a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t};
    }

    RGB scale(const RGB &a, float s)
    {
      return {a.r * s, a.g * s, a.b * s};
    }

    /* A color of the rainbow, @p h in [0, 1): */
    RGB hue(float h)
    {
      h = (h - std::floor(h)) * 6.f;
      const float x = 1.f - std::abs(std::fmod(h, 2.f) - 1.f);
      const std::array<RGB, 6> sextants = {
          RGB{1, x, 0}, {x, 1, 0}, {0, 1, x}, {0, x, 1}, {x, 0, 1}, {1, 0, x}};
      return scale(sextants[static_cast<int>(h) % 6], 255.f);
    }

    Color to_color(const RGB &c)
    {
      const auto byte = [](float v) {
        return static_cast<std::uint8_t>(std::clamp(v, 0.f, 255.f));
      };
      return Color::RGB(byte(c.r), byte(c.g), byte(c.b));
    }

    const RGB white = {255, 255, 255};
    const RGB black = {0, 0, 0};
    const RGB red = {255, 60, 60};
    const RGB yellow = {255, 230, 60};
    const RGB neon = {255, 40, 220};
    const RGB phosphor = {80, 255, 120};


    float ease_out_bounce(float t)
    {
      constexpr float n = 7.5625f;
      constexpr float d = 2.75f;
      if (t < 1.f / d)
        return n * t * t;
      if (t < 2.f / d) {
        t -= 1.5f / d;
        return n * t * t + 0.75f;
      }
      if (t < 2.5f / d) {
        t -= 2.25f / d;
        return n * t * t + 0.9375f;
      }
      t -= 2.625f / d;
      return n * t * t + 0.984375f;
    }


    /*
     * A pixel font of 5x7 pixels:
     */

    using Glyph = std::array<std::string_view, 7>;

    const Glyph &glyph(char c)
    {
      // clang-format off
      static const std::map<char, Glyph> font = {
        {'A', {".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"}},
        {'B', {"####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."}},
        {'C', {".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."}},
        {'D', {"####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."}},
        {'E', {"#####", "#....", "#....", "####.", "#....", "#....", "#####"}},
        {'F', {"#####", "#....", "#....", "####.", "#....", "#....", "#...."}},
        {'G', {".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".####"}},
        {'H', {"#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"}},
        {'I', {".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."}},
        {'J', {"..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."}},
        {'K', {"#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"}},
        {'L', {"#....", "#....", "#....", "#....", "#....", "#....", "#####"}},
        {'M', {"#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"}},
        {'N', {"#...#", "#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#"}},
        {'O', {".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."}},
        {'P', {"####.", "#...#", "#...#", "####.", "#....", "#....", "#...."}},
        {'Q', {".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"}},
        {'R', {"####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"}},
        {'S', {".####", "#....", "#....", ".###.", "....#", "....#", "####."}},
        {'T', {"#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."}},
        {'U', {"#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."}},
        {'V', {"#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."}},
        {'W', {"#...#", "#...#", "#...#", "#.#.#", "#.#.#", "#.#.#", ".#.#."}},
        {'X', {"#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"}},
        {'Y', {"#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."}},
        {'Z', {"#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"}},
        {'0', {".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."}},
        {'1', {"..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."}},
        {'2', {".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"}},
        {'3', {"#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."}},
        {'4', {"...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."}},
        {'5', {"#####", "#....", "####.", "....#", "....#", "#...#", ".###."}},
        {'6', {"..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."}},
        {'7', {"#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."}},
        {'8', {".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."}},
        {'9', {".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."}},
        {'.', {".....", ".....", ".....", ".....", ".....", ".##..", ".##.."}},
        {',', {".....", ".....", ".....", ".....", ".##..", "..#..", ".#..."}},
        {'!', {"..#..", "..#..", "..#..", "..#..", "..#..", ".....", "..#.."}},
        {'?', {".###.", "#...#", "....#", "...#.", "..#..", ".....", "..#.."}},
        {'-', {".....", ".....", ".....", ".###.", ".....", ".....", "....."}},
        {'+', {".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."}},
        {'*', {".....", "#.#.#", ".###.", "#####", ".###.", "#.#.#", "....."}},
        {':', {".....", ".##..", ".##..", ".....", ".##..", ".##..", "....."}},
        {'/', {"....#", "....#", "...#.", "..#..", ".#...", "#....", "#...."}},
        {'(', {"...#.", "..#..", ".#...", ".#...", ".#...", "..#..", "...#."}},
        {')', {".#...", "..#..", "...#.", "...#.", "...#.", "..#..", ".#..."}},
        {'\'', {"..#..", "..#..", ".#...", ".....", ".....", ".....", "....."}},
      };
      // clang-format on
      static const Glyph blank = {
          ".....", ".....", ".....", ".....", ".....", ".....", "....."};

      const auto it = font.find(
          static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
      return it == font.end() ? blank : it->second;
    }


    /*
     * A frame buffer with two pixels per character, drawn with the upper
     * half block "▀", and text on top:
     */
    class Canvas
    {
    public:
      Canvas(int columns, int rows)
          : columns_(columns)
          , rows_(rows)
          , pixels_(2 * columns * rows)
          , cells_(columns * rows)
      {
      }

      /* The width in pixels, the same as the number of columns: */
      int width() const
      {
        return columns_;
      }

      int height() const
      {
        return 2 * rows_;
      }

      int rows() const
      {
        return rows_;
      }

      void clear()
      {
        std::ranges::fill(pixels_, black);
        std::ranges::fill(cells_, Cell{});
      }

      RGB get(int x, int y) const
      {
        if (x < 0 || y < 0 || x >= width() || y >= height())
          return black;
        return pixels_[y * columns_ + x];
      }

      void blend(int x, int y, const RGB &color, float alpha = 1.f)
      {
        if (x < 0 || y < 0 || x >= width() || y >= height())
          return;
        auto &pixel = pixels_[y * columns_ + x];
        pixel = mix(pixel, color, alpha);
      }

      void blend(float x, float y, const RGB &color, float alpha = 1.f)
      {
        blend(static_cast<int>(std::floor(x)),
              static_cast<int>(std::floor(y)),
              color,
              alpha);
      }

      void line(float x0, float y0, float x1, float y1, const RGB &color)
      {
        const int steps = std::clamp(
            static_cast<int>(std::max(std::abs(x1 - x0), std::abs(y1 - y0))),
            1,
            400);
        for (int i = 0; i <= steps; ++i) {
          const float s = static_cast<float>(i) / steps;
          blend(x0 + (x1 - x0) * s, y0 + (y1 - y0) * s, color);
        }
      }

      /* A disc, @p shade maps the position relative to the center and
       * the radius to a color: */
      template <typename F>
      void disc(float cx, float cy, float r, const F &shade)
      {
        for (int y = static_cast<int>(cy - r); y <= cy + r; ++y)
          for (int x = static_cast<int>(cx - r); x <= cx + r; ++x) {
            const float dx = (x + 0.5f - cx) / r;
            const float dy = (y + 0.5f - cy) / r;
            if (dx * dx + dy * dy <= 1.f)
              blend(x, y, shade(dx, dy));
          }
      }

      /* Blend the whole frame with @p color: */
      void fade(const RGB &color, float alpha)
      {
        for (auto &pixel : pixels_)
          pixel = mix(pixel, color, alpha);
      }

      /* Black out everything outside of the circle around (cx, cy): */
      void iris(float cx, float cy, float r)
      {
        for (int y = 0; y < height(); ++y)
          for (int x = 0; x < width(); ++x)
            if (std::hypot(x - cx, y - cy) > r)
              pixels_[y * columns_ + x] = black;

        for (int row = 0; row < rows_; ++row)
          for (int column = 0; column < columns_; ++column)
            if (std::hypot(column - cx, 2 * row - cy) > r)
              cells_[row * columns_ + column] = Cell{};
      }

      void text(int column,
                int row,
                const std::string &text,
                const RGB &color,
                bool bold = false)
      {
        if (row < 0 || row >= rows_)
          return;
        for (const auto &character : ftxui::Utf8ToGlyphs(text)) {
          if (column >= 0 && column < columns_)
            cells_[row * columns_ + column] = {character, color, bold, true};
          ++column;
        }
      }

      /* Text centered in the row @p row: */
      void centered(int row,
                    const std::string &text,
                    const RGB &color,
                    bool bold = false)
      {
        const int width = ftxui::string_width(text);
        this->text((columns_ - width) / 2, row, text, color, bold);
      }

      /* Text in the pixel font, scaled by @p s, with a shadow: */
      void print(int x, int y, std::string_view text, const RGB &color, int s)
      {
        for (const char c : text) {
          const auto &rows = glyph(c);
          for (int gy = 0; gy < 7; ++gy)
            for (int gx = 0; gx < 5; ++gx)
              if (rows[gy][gx] == '#')
                for (int i = 0; i < s; ++i)
                  for (int j = 0; j < s; ++j) {
                    blend(x + gx * s + i + 1, y + gy * s + j + 1, black, 0.8f);
                    blend(x + gx * s + i, y + gy * s + j, color);
                  }
          x += 6 * s;
        }
      }

      /* Render the frame, shifted down by @p shake pixels: */
      Screen render(int shake) const
      {
        Screen screen(columns_, rows_);
        for (int row = 0; row < rows_; ++row)
          for (int column = 0; column < columns_; ++column) {
            const RGB top = get(column, 2 * row - shake);
            const RGB bottom = get(column, 2 * row + 1 - shake);
            const auto &cell = cells_[row * columns_ + column];

            auto &pixel = screen.PixelAt(column, row);
            if (cell.used) {
              pixel.character = cell.character;
              pixel.foreground_color = to_color(cell.color);
              pixel.background_color =
                  to_color(scale(mix(top, bottom, 0.5f), 0.3f));
              pixel.bold = cell.bold;
            } else {
              pixel.character = "▀";
              pixel.foreground_color = to_color(top);
              pixel.background_color = to_color(bottom);
            }
          }
        return screen;
      }

    private:
      struct Cell {
        std::string character;
        RGB color;
        bool bold = false;
        bool used = false;
      };

      int columns_;
      int rows_;
      std::vector<RGB> pixels_;
      std::vector<Cell> cells_;
    };


    /*
     * The intro:
     */

    /* The first @p n characters of @p text: */
    std::string typed(const std::string &text, float n)
    {
      const auto glyphs = ftxui::Utf8ToGlyphs(text);
      const auto count = std::clamp(static_cast<int>(n), 0, int(glyphs.size()));
      return std::accumulate(
          glyphs.begin(), glyphs.begin() + count, std::string());
    }


    class Intro
    {
    public:
      /* When the hyperspace ends and the title appears: */
      static constexpr float title = 2.5f;

      /* When "PRESS START" appears and a key press is accepted: */
      static constexpr float press_start = title + 2.6f;

      Intro(int columns, int rows, std::string stage, unsigned int n_ranks)
          : canvas_(columns, rows)
          , stage_(std::move(stage))
          , n_ranks_(n_ranks)
      {
        const int w = canvas_.width();
        const int h = canvas_.height();

        logo_scale_ = std::clamp((w - 6) / 35, 2, 3);
        logo_x_ = (w - 35 * logo_scale_) / 2;
        logo_y_ = 6;

        /* Below the logo, the kanji and the subtitle: */
        text_row_ = (logo_y_ + 7 * logo_scale_) / 2 + 1;
        horizon_ = std::max(h * 11 / 20, 2 * (text_row_ + 3));
        sun_radius_ = std::min(0.8f * (horizon_ - logo_y_), 0.2f * w);
        scroller_amplitude_ = std::clamp((h - horizon_) / 8.f, 1.f, 4.f);
        scroller_base_ = h - 10.f - scroller_amplitude_;

        std::mt19937 random(1985);
        std::uniform_real_distribution<float> uniform(0.f, 1.f);
        for (auto &star : warp_stars_)
          star = {(2.f * uniform(random) - 1.f) * w / h * 2.f,
                  2.f * uniform(random) - 1.f,
                  uniform(random)};
        for (auto &star : sky_stars_)
          star = {uniform(random) * w,
                  uniform(random) * horizon_,
                  uniform(random) * 6.3f};
      }

      Screen frame(float t)
      {
        canvas_.clear();
        hud(t);

        if (t < title) {
          warp(t);
          return canvas_.render(0);
        }

        landscape(t);
        dragon(t);
        logo(t);
        scroller(t);
        texts(t);

        if (t < title + 0.3f)
          canvas_.fade(white, 1.f - (t - title) / 0.3f);

        if (t > start_ + 0.5f) {
          const float s = std::clamp((t - start_ - 0.5f) / 0.8f, 0.f, 1.f);
          const float r =
              std::hypot(float(canvas_.width()), float(canvas_.height())) / 2;
          canvas_.iris(canvas_.width() / 2.f,
                       canvas_.height() / 2.f,
                       r * std::pow(1.f - s, 1.5f));
        }

        /* The ground shakes when the logo lands: */
        const bool shake = t > title + 1.2f && t < title + 1.4f;
        return canvas_.render(shake ? static_cast<int>(t * 40) % 2 : 0);
      }

      /* Start the game at time @p t, the first call counts: */
      void start(float t)
      {
        start_ = std::min(start_, t);
      }

      /* Whether the intro is over: */
      bool finished(float t) const
      {
        return t > start_ + 1.3f;
      }

    private:
      /* The score line at the top: */
      void hud(float t)
      {
        const int columns = canvas_.width();
        const auto score =
            std::format("{:06}", static_cast<long>(t * 4321) * 10 % 1000000);
        const auto credits = std::format("{:02}", std::min(n_ranks_, 99u));

        if (static_cast<int>(t * 2.5f) % 2 == 0)
          canvas_.text(2, 0, "1UP", red, true);
        canvas_.text(6, 0, score, white);

        canvas_.text(columns / 2 - 8, 0, "HI-SCORE", red, true);
        canvas_.text(columns / 2 + 1, 0, "314159", white);

        canvas_.text(columns - 11, 0, "CREDIT", red, true);
        canvas_.text(columns - 4, 0, credits, white);
      }

      /* Hyperspace: stars rushing by, faster and faster: */
      void warp(float t)
      {
        const float cx = canvas_.width() / 2.f;
        const float cy = canvas_.height() / 2.f;
        const float f = canvas_.height() / 4.f;

        const auto travel = [](float t) { return 0.3f * t + 0.6f * t * t * t; };
        const float step = travel(t) - travel(t - 0.06f);

        for (const auto &star : warp_stars_) {
          float z = std::fmod(star[2] - travel(t), 1.f);
          z = (z < 0.f ? z + 1.f : z) + 0.02f;
          const float previous = z + step;

          const float x = cx + star[0] / z * f;
          const float y = cy + star[1] / z * f;
          const float brightness = std::pow(1.f - z, 2.f);
          const RGB color =
              scale(mix({120, 160, 255}, white, 1.f - z), brightness);

          if (previous < 1.02f)
            canvas_.line(cx + star[0] / previous * f,
                         cy + star[1] / previous * f,
                         x,
                         y,
                         color);
          else
            canvas_.blend(x, y, color);
        }

        const int row = canvas_.rows() / 2;
        const bool cursor = static_cast<int>(t * 6) % 2 == 0;
        const auto line = typed("THE RYUJIN AUTHORS", (t - 0.2f) * 22.f);
        const auto present = typed("PRESENT", (t - 1.1f) * 16.f);
        canvas_.centered(row - 1,
                         line + (present.empty() && cursor ? "█" : " "),
                         phosphor,
                         true);
        if (t > 1.1f)
          canvas_.centered(
              row + 1, present + (cursor ? "█" : " "), phosphor, true);
      }

      /* The sky, the sun, the mountains, and a grid floor: */
      void landscape(float t)
      {
        const int w = canvas_.width();
        const int h = canvas_.height();
        const float cx = w / 2.f;

        for (int y = 0; y < horizon_; ++y) {
          const float v = static_cast<float>(y) / horizon_;
          const RGB sky = mix(mix({8, 0, 24}, {70, 10, 90}, v),
                              {255, 70, 140},
                              std::pow(v, 6.f));
          for (int x = 0; x < w; ++x)
            canvas_.blend(x, y, sky);
        }

        for (const auto &[x, y, phase] : sky_stars_) {
          const float twinkle = 0.5f + 0.5f * std::sin(3.f * t + phase);
          canvas_.blend(x, y, white, twinkle * (1.f - y / horizon_));
        }

        /* A striped sun on the horizon, the stripes moving down: */
        const float r = sun_radius_;
        for (int y = static_cast<int>(horizon_ - r); y < horizon_; ++y) {
          const float v = (y - (horizon_ - r)) / r;
          if (v > 0.4f) {
            const int gap = 1 + static_cast<int>((v - 0.4f) * 6.f);
            const int phase = static_cast<int>(std::floor(y - 8.f * t));
            if (((phase % 6) + 6) % 6 < gap)
              continue;
          }
          const float half = std::sqrt(r * r - (horizon_ - y) * (horizon_ - y));
          for (int x = static_cast<int>(cx - half); x <= cx + half; ++x)
            canvas_.blend(x, y, mix({255, 235, 90}, {255, 40, 150}, v));
        }

        /* Mountains, lower in front of the sun: */
        for (int x = 0; x < w; ++x) {
          const float open = std::min(1.f, std::abs(x - cx) / (1.3f * r));
          const float m = 0.45f * r * (0.4f + 0.6f * open) *
                          (0.55f + 0.25f * std::sin(0.09f * x + 0.5f) +
                           0.2f * std::sin(0.23f * x + 2.f));
          const int top = horizon_ - static_cast<int>(m);
          for (int y = top; y < horizon_; ++y)
            canvas_.blend(
                x, y, mix({60, 10, 90}, {25, 0, 45}, (y - top) / (m + 1.f)));
          canvas_.blend(x, top, {255, 80, 200}, 0.7f);
        }

        /* The grid floor, rushing towards the viewer: */
        for (int y = horizon_; y < h; ++y) {
          const float d = y - horizon_ + 1.f;
          const float z = 16.f / d;
          const float z_next = 16.f / (d + 1.f);
          const float fog =
              std::pow(std::clamp(d / (0.5f * (h - horizon_)), 0.f, 1.f), 1.5f);
          const bool across =
              std::floor(z + 2.5f * t) != std::floor(z_next + 2.5f * t);

          const RGB ground =
              mix({30, 0, 50}, {8, 0, 20}, (d - 1.f) / (h - horizon_));
          for (int x = 0; x < w; ++x) {
            const float a = (x - cx) * z / 8.f;
            const float b = (x + 1 - cx) * z / 8.f;
            const bool along = std::floor(a) != std::floor(b);
            canvas_.blend(x, y, ground);
            if (across || along)
              canvas_.blend(x, y, neon, fog);
          }
        }
        for (int x = 0; x < w; ++x)
          canvas_.blend(x, horizon_, {255, 140, 230});
      }

      /*
       * The position of the dragon (and the pearl it chases) at time τ,
       * flying from the right to the left once every seven seconds:
       */
      std::pair<float, float> path(float tau) const
      {
        tau = std::fmod(tau, 7.f);
        tau = tau < 0.f ? tau + 7.f : tau;

        const float w = canvas_.width();
        const float speed = (w + 150.f) / 4.f;
        /* Below the logo, with a wave of at least 3 pixels: */
        const float logo_bottom = logo_y_ + 7.f * logo_scale_;
        const float base =
            std::max(horizon_ - 0.55f * sun_radius_, logo_bottom + 4.f);
        const float amplitude = std::max(
            3.f, std::min(0.45f * sun_radius_, base - logo_bottom - 4.f));
        return {w + 12.f - speed * tau,
                base + amplitude * (std::sin(2.4f * tau) +
                                    0.3f * std::sin(5.1f * tau))};
      }

      /*
       * The unit tangent (dx, dy) of the path at τ, and the unit normal
       * (nx, ny) pointing up:
       */
      std::array<float, 4> direction(float tau) const
      {
        const auto [x, y] = path(tau);
        const auto [xf, yf] = path(tau + 0.01f);
        const float length = std::hypot(xf - x, yf - y) + 1e-6f;
        const float dx = (xf - x) / length;
        const float dy = (yf - y) / length;
        return dx < 0.f ? std::array{dx, dy, -dy, dx}
                        : std::array{dx, dy, dy, -dx};
      }

      /* Ryujin, the dragon god, chasing the flaming pearl: */
      void dragon(float t)
      {
        const float tau = t - title - 0.2f;
        if (tau < 0.f)
          return;

        /* The pearl, glowing: */
        const auto [px, py] = path(tau + 0.3f);
        const float glow = 7.f + std::sin(20.f * t);
        for (int y = static_cast<int>(py - glow); y <= py + glow; ++y)
          for (int x = static_cast<int>(px - glow); x <= px + glow; ++x) {
            const float d = std::hypot(x + 0.5f - px, y + 0.5f - py) / glow;
            if (d < 1.f)
              canvas_.blend(x, y, {255, 90, 30}, 0.8f * (1.f - d) * (1.f - d));
          }
        canvas_.disc(px, py, 2.2f, [](float dx, float dy) {
          return mix({255, 250, 220}, {255, 150, 60}, std::hypot(dx, dy));
        });

        /* The body, from the tail to the head: */
        constexpr int n = 44;
        for (int k = n - 1; k >= 1; --k) {
          const auto [x, y] = path(tau - 0.045f * k);
          const auto [dx, dy, nx, ny] = direction(tau - 0.045f * k);

          const float s = static_cast<float>(k) / n;
          const float r = 3.2f * (1.f - 0.75f * s * s);
          const bool ring = k % 5 == 0;
          canvas_.disc(x, y, r, [ring](float, float dy) {
            if (dy > 0.4f)
              return RGB{230, 190, 90};
            const RGB scales = ring ? RGB{240, 190, 60} : RGB{40, 190, 110};
            return mix(scales, {200, 255, 210}, -0.5f * dy);
          });

          /* Fins on the back, claws at the legs: */
          if (k % 2 == 0 && k < n - 4) {
            canvas_.blend(
                x + nx * (r + 1.f), y + ny * (r + 1.f), {230, 50, 40});
            if (k % 4 == 0)
              canvas_.blend(
                  x + nx * (r + 2.f), y + ny * (r + 2.f), {255, 90, 50});
          }
          if (k == 9 || k == 27)
            for (int i = 1; i <= 3; ++i) {
              const float cx = x - nx * (r + i) - 0.5f * i;
              const float cy = y - ny * (r + i);
              canvas_.blend(cx, cy, {240, 190, 60});
              if (i == 3) {
                canvas_.blend(cx - 1.f, cy, {255, 240, 200});
                canvas_.blend(cx + 1.f, cy, {255, 240, 200});
              }
            }
        }

        /* The head, facing the pearl: */
        const auto [x, y] = path(tau);
        const auto [dx, dy, nx, ny] = direction(tau);

        const auto green = [](float, float dy) {
          return mix({40, 190, 110}, {200, 255, 210}, -0.5f * dy);
        };
        /* Mane and horns behind the head: */
        for (int i = 0; i < 5; ++i)
          canvas_.disc(x - dx * (2.f + i) + nx * (2.f + 0.6f * i),
                       y - dy * (2.f + i) + ny * (2.f + 0.6f * i),
                       1.3f,
                       [](float, float) {
                         return RGB{230, 40, 40};
                       });
        canvas_.line(x + nx * 3.f,
                     y + ny * 3.f,
                     x - dx * 5.f + nx * 7.f,
                     y - dy * 5.f + ny * 7.f,
                     {240, 200, 80});
        canvas_.line(x - dx * 1.f + nx * 3.f,
                     y - dy * 1.f + ny * 3.f,
                     x - dx * 7.f + nx * 5.f,
                     y - dy * 7.f + ny * 5.f,
                     {240, 200, 80});

        canvas_.disc(x, y, 3.8f, green);
        canvas_.disc(x + dx * 2.5f, y + dy * 2.5f, 3.f, green);
        canvas_.disc(x + dx * 5.f, y + dy * 5.f, 2.1f, green);
        canvas_.disc(x + dx * 7.f, y + dy * 7.f, 1.5f, green);

        /* The open mouth, the nose, and the eye: */
        canvas_.line(x + dx * 3.f - nx,
                     y + dy * 3.f - ny,
                     x + dx * 8.f - nx * 1.5f,
                     y + dy * 8.f - ny * 1.5f,
                     {120, 0, 20});
        canvas_.blend(x + dx * 8.2f, y + dy * 8.2f, {255, 60, 60});
        canvas_.blend(
            x + dx * 1.5f + nx * 1.5f, y + dy * 1.5f + ny * 1.5f, white);
        canvas_.blend(x + dx * 2.5f + nx * 1.5f,
                      y + dy * 2.5f + ny * 1.5f,
                      {255, 40, 40});

        /* Whiskers, waving: */
        for (const float side : {-1.f, 1.f})
          for (int i = 0; i < 12; ++i) {
            const float s = static_cast<float>(i);
            const float wave = 1.5f * std::sin(0.8f * s + 8.f * t);
            const float ox =
                dx * (7.f - 1.2f * s) + nx * side * (1.f + 0.5f * s + wave);
            const float oy =
                dy * (7.f - 1.2f * s) + ny * side * (1.f + 0.5f * s + wave);
            canvas_.blend(x + ox, y + oy, {255, 170, 60}, 1.f - s / 14.f);
          }
      }

      /* The logo, in chrome, dropping in letter by letter: */
      void logo(float t)
      {
        const int w = canvas_.width();
        const int h = canvas_.height();
        const int s = logo_scale_;
        const std::string_view name = "RYUJIN";

        /* The height within the letter of all pixels of the logo: */
        std::vector<float> mask(w * h, -1.f);
        for (std::size_t i = 0; i < name.size(); ++i) {
          const float start = title + 0.1f * i;
          if (t < start)
            continue;
          const float f = std::min(1.f, (t - start) / 0.7f);
          const int top =
              logo_y_ - static_cast<int>((logo_y_ + 7 * s + 4) *
                                         (1.f - ease_out_bounce(f)));
          const int left = logo_x_ + 6 * s * static_cast<int>(i);

          const auto &rows = glyph(name[i]);
          for (int y = 0; y < 7 * s; ++y)
            for (int x = 0; x < 5 * s; ++x)
              if (rows[y / s][x / s] == '#' && top + y >= 0 && top + y < h &&
                  left + x < w)
                mask[(top + y) * w + left + x] = (y + 0.5f) / (7 * s);
        }

        const auto inside = [&](int x, int y) {
          return x >= 0 && y >= 0 && x < w && y < h && mask[y * w + x] >= 0.f;
        };

        const float shine =
            std::fmod((t - title - 1.3f) * 90.f, 35.f * s + 200.f) - 40.f;
        for (int y = 0; y < h; ++y)
          for (int x = 0; x < w; ++x) {
            if (inside(x - 2, y - 2) && !inside(x, y))
              canvas_.blend(x, y, black, 0.7f);
          }
        for (int y = 0; y < h; ++y)
          for (int x = 0; x < w; ++x) {
            if (!inside(x, y)) {
              if (inside(x - 1, y) || inside(x + 1, y) || inside(x, y - 1) ||
                  inside(x, y + 1))
                canvas_.blend(x, y, {30, 0, 60});
              continue;
            }

            /* Chrome: a sky above and a desert below a dark horizon: */
            const float v = mask[y * w + x];
            RGB color =
                v < 0.5f
                    ? mix({90, 170, 255}, {240, 250, 255}, v / 0.5f)
                    : mix({120, 60, 20}, {255, 200, 90}, (v - 0.5f) / 0.5f);
            if (std::abs(v - 0.5f) < 0.5f / (7 * s))
              color = {60, 20, 40};

            const float u = x - logo_x_ - 0.6f * (y - logo_y_) - shine;
            if (t > title + 1.3f && u >= 0.f && u < 3.f)
              color = mix(color, white, 0.85f);
            canvas_.blend(x, y, color);
          }

        /* Glints at the corners of the logo: */
        if (t > title + 1.3f) {
          const std::array<std::string, 6> sparkle = {
              "·", "+", "✦", "+", "·", " "};
          const int bottom = (logo_y_ + 7 * s) / 2;
          const int right = logo_x_ + 35 * s;
          const std::array<std::pair<int, int>, 3> spots = {
              {{logo_x_ - 1, logo_y_ / 2 - 1},
               {right, logo_y_ / 2},
               {logo_x_ + 17 * s, bottom}}};
          for (std::size_t k = 0; k < spots.size(); ++k) {
            const auto frame = (static_cast<int>(t * 12) + 3 * k) % 8;
            if (frame < sparkle.size())
              canvas_.text(
                  spots[k].first, spots[k].second, sparkle[frame], white, true);
          }
        }
      }

      /* A sine scroller of greetings above the bottom line: */
      void scroller(float t)
      {
        if (t < title + 0.5f)
          return;

        const std::string_view message =
            "*** GREETINGS TO DEAL.II * KOKKOS * FTXUI * MPI ***   "
            "INSERT COIN TO SOLVE CONSERVATION EQUATIONS   ***";
        const float offset = 45.f * (t - title - 0.5f);
        const float length = 6.f * message.size();

        for (std::size_t i = 0; i < message.size(); ++i) {
          /* Over and over again, once the message has come in: */
          const float d = offset - 6.f * i;
          const float x =
              canvas_.width() - (d < 0.f ? d : std::fmod(d, length));
          if (x < -6.f || x > canvas_.width())
            continue;
          const float y = scroller_base_ +
                          scroller_amplitude_ * std::sin(0.09f * x + 6.f * t);
          canvas_.print(static_cast<int>(x),
                        static_cast<int>(y),
                        message.substr(i, 1),
                        hue(0.004f * x + 0.5f * t),
                        1);
        }
      }

      /* The kanji, the subtitle, the stage, and the credits: */
      void texts(float t)
      {
        const int rows = canvas_.rows();

        if (t > title + 1.2f) {
          canvas_.centered(text_row_, "─── 龍 神 ───", {255, 70, 50}, true);

          std::string subtitle = "HIGH-PERFORMANCE FINITE-ELEMENT SOLVER FOR "
                                 "CONSERVATION EQUATIONS";
          if (static_cast<int>(subtitle.size()) + 4 > canvas_.width())
            subtitle = "SOLVER FOR CONSERVATION EQUATIONS";
          canvas_.centered(text_row_ + 1,
                           typed(subtitle, (t - title - 1.25f) * 70.f),
                           {120, 230, 255});

          canvas_.centered(
              rows - 1,
              "© 2026 THE RYUJIN AUTHORS  ·  APACHE-2.0 WITH LLVM-EXCEPTION",
              {150, 130, 200});
        }

        /*
         * The stage and the call to start on the floor, in the pixel font
         * on a dark band if there is enough space:
         */
        const int space =
            static_cast<int>(scroller_base_ - scroller_amplitude_) - horizon_;
        const auto announce =
            [&](int line, const std::string &text, RGB color) {
              const int width = 6 * static_cast<int>(text.size()) - 1;
              if (space < 24 || width + 6 > canvas_.width()) {
                const int row = horizon_ / 2 + 1 + line;
                canvas_.centered(row, text, color, true);
                return;
              }
              const int x = (canvas_.width() - width) / 2;
              const int y = horizon_ + (space - 17) / 2 + 10 * line;
              for (int j = -2; j < 9; ++j)
                for (int i = -3; i < width + 3; ++i)
                  canvas_.blend(x + i, y + j, black, 0.6f);
              canvas_.print(x, y, text, color, 1);
            };

        if (t > press_start)
          announce(0, stage_, white);
        if (t > press_start && t < start_ && static_cast<int>(t * 5) % 2 == 0)
          announce(1, "PRESS START", yellow);
        if (t >= start_)
          announce(1, "GET READY!", hue(3.f * t));
      }

      Canvas canvas_;
      std::string stage_;
      unsigned int n_ranks_;

      int logo_scale_;
      int logo_x_;
      int logo_y_;
      int text_row_;
      int horizon_;
      float sun_radius_;
      float scroller_amplitude_;
      float scroller_base_;

      /* The time of the key press: */
      float start_ = std::numeric_limits<float>::infinity();

      std::array<std::array<float, 3>, 160> warp_stars_;
      std::array<std::array<float, 3>, 70> sky_stars_;
    };


    /*
     * The logo:
     */

    /* The width of the logo scaled to a height of @p height pixels: */
    int logo_width(int height)
    {
      return static_cast<int>(splash::logo_pixels[0].size()) * height /
             static_cast<int>(splash::logo_pixels.size());
    }


    /*
     * The logo scaled to a height of @p height pixels, row by row: average
     * the opaque pixels of each block, a block with less than half of its
     * pixels opaque is transparent:
     */
    std::vector<std::optional<RGB>> scaled_logo(int height)
    {
      using splash::logo_palette;
      using splash::logo_pixels;
      constexpr std::string_view digits =
          "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
      const int n = static_cast<int>(logo_pixels.size());
      const int m = static_cast<int>(logo_pixels[0].size());
      const int width = logo_width(height);

      std::vector<std::optional<RGB>> pixels(width * height);
      for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
          RGB sum;
          int opaque = 0;
          int total = 0;
          for (int j = y * n / height; j < (y + 1) * n / height; ++j)
            for (int i = x * m / width; i < (x + 1) * m / width; ++i) {
              ++total;
              const auto index = digits.find(logo_pixels[j][i]);
              if (index == std::string_view::npos)
                continue;
              const auto c = logo_palette[index];
              sum.r += static_cast<float>(c >> 16);
              sum.g += static_cast<float>((c >> 8) & 0xff);
              sum.b += static_cast<float>(c & 0xff);
              ++opaque;
            }
          if (opaque > 0 && 2 * opaque >= total)
            pixels[y * width + x] = scale(sum, 1.f / opaque);
        }
      return pixels;
    }


    /*
     * Terminal access:
     */

    /* Switch to the alternate screen and hide the cursor, and back: */
    constexpr std::string_view enter = "\033[?1049h\033[?25l\033[2J";
    constexpr std::string_view leave = "\033[0m\033[?25h\033[?1049l";

    /* The terminal and the keyboard of the intro, for restore(): */
    const terminal::Output *output = nullptr;
    const terminal::Keyboard *keyboard = nullptr;

    /* Leave the alternate screen if interrupted: */
    void restore(int signal)
    {
      keyboard->restore();
      output->write(leave);
      std::signal(signal, SIG_DFL);
      std::raise(signal);
    }


    /*
     * The splash screen of SplashScreen::logo: the logo, the name, the
     * version, the run, and the copyright centered on a cleared screen,
     * with the cursor placed such that the following lines of output
     * appear at the bottom:
     */
    void play_logo(const std::string &equation,
                   int dim,
                   unsigned int n_ranks,
                   int columns,
                   int rows)
    {
      /* The number of lines left free at the bottom: */
      constexpr int reserve = 4;

      struct Segment {
        std::string text;
        bool bold = false;
        bool dim = false;
      };
      using Line = std::vector<Segment>;

      const std::string version = version_string();
      const std::vector<Line> lines = {
          {{"ryujin", true}, {version.substr(version.find(' '))}},
          {{"High-performance finite-element solver for conservation "
            "equations",
            false,
            true}},
          {},
          {{std::format("{} · {}D · {} MPI rank{}",
                        equation,
                        dim,
                        n_ranks,
                        n_ranks == 1 ? "" : "s")}},
          {},
          {{"© 2026 by the ryujin authors", false, true}},
          {{"Apache-2.0 WITH LLVM-exception", false, true}},
          {{"https://conservation-laws.org/ryujin/", false, true}},
      };
      const int n_lines = static_cast<int>(lines.size());

      /*
       * The largest logo that fits, if any, with a blank line above the
       * free lines at the bottom:
       */
      int logo_height = 0;
      for (const int candidate : {64, 48, 32, 24})
        if (rows >= candidate / 2 + 1 + n_lines + 1 + reserve &&
            columns >= logo_width(candidate)) {
          logo_height = candidate;
          break;
        }
      if (logo_height == 0 && rows < n_lines + 1 + reserve)
        return;

      const int logo_rows = logo_height / 2;
      const int height = logo_height > 0 ? logo_rows + 1 + n_lines : n_lines;
      Screen screen(columns, height);

      /*
       * The logo in half blocks, transparent pixels show the background of
       * the terminal:
       */
      int row = 0;
      if (logo_height > 0) {
        const auto logo = scaled_logo(logo_height);
        const int width = logo_width(logo_height);
        const int left = (columns - width) / 2;
        for (; row < logo_rows; ++row)
          for (int x = 0; x < width; ++x) {
            const auto &top = logo[2 * row * width + x];
            const auto &bottom = logo[(2 * row + 1) * width + x];
            auto &pixel = screen.PixelAt(left + x, row);
            if (top) {
              pixel.character = "▀";
              pixel.foreground_color = to_color(*top);
              if (bottom)
                pixel.background_color = to_color(*bottom);
            } else if (bottom) {
              pixel.character = "▄";
              pixel.foreground_color = to_color(*bottom);
            }
          }
        ++row;
      }

      for (const auto &line : lines) {
        int width = 0;
        for (const auto &segment : line)
          width += ftxui::string_width(segment.text);
        int column = (columns - width) / 2;
        for (const auto &segment : line)
          for (const auto &character : ftxui::Utf8ToGlyphs(segment.text)) {
            auto &pixel = screen.PixelAt(column++, row);
            pixel.character = character;
            pixel.bold = segment.bold;
            pixel.dim = segment.dim;
          }
        ++row;
      }

      /*
       * Scroll the previous output off the screen, then center the splash
       * screen, but above the free lines at the bottom:
       */
      const int top =
          std::min((rows - height) / 2, rows - reserve - 1 - height);
      std::cout << std::string(rows, '\n') << "\033[" << top + 1 << ";1H"
                << screen.ToString() << "\033[0m\033[" << rows - reserve + 1
                << ";1H" << std::flush;
    }


    /* The intro of SplashScreen::arcade: */
    void play_arcade(const terminal::Output &tty,
                     const std::string &equation,
                     int dim,
                     unsigned int n_ranks,
                     int columns,
                     int rows)
    {
      std::string stage = std::format("STAGE {}D - {}", dim, equation);
      std::ranges::transform(stage, stage.begin(), ::toupper);
      Intro intro(columns, rows, stage, n_ranks);

      std::cout << std::flush;
      const terminal::Keyboard keys;
      output = &tty;
      keyboard = &keys;
      const auto previous_int = std::signal(SIGINT, restore);
      const auto previous_term = std::signal(SIGTERM, restore);
      tty.write(enter);

      using clock = std::chrono::steady_clock;
      const auto start = clock::now();
      for (auto next = start;; next += std::chrono::milliseconds(50)) {
        const float t =
            std::chrono::duration<float>(clock::now() - start).count();
        if (intro.finished(t))
          break;
        /* Show "PRESS START" for a moment, also for an early key press: */
        if (t > Intro::press_start && (!keys.may_wait() || keys.key_pressed()))
          intro.start(std::max(t, Intro::press_start + 0.6f));
        tty.write("\033[H" + intro.frame(t).ToString());
        std::this_thread::sleep_until(next + std::chrono::milliseconds(50));
      }

      keys.restore();
      tty.write(leave);
      std::signal(SIGINT, previous_int);
      std::signal(SIGTERM, previous_term);
      output = nullptr;
      keyboard = nullptr;
    }
  } // namespace


  void play_splash_screen(const std::string &equation,
                          int dim,
                          unsigned int n_ranks,
                          SplashScreen splash_screen)
  {
    if (splash_screen == SplashScreen::none ||
        !terminal::stdout_reaches_terminal())
      return;

    const terminal::Output tty;
    const auto size = tty.size();
    if (!tty.controlling_terminal() || !size || size->first < 80)
      return;

    const auto [columns, rows] = *size;
    if (splash_screen == SplashScreen::logo)
      play_logo(equation, dim, n_ranks, columns, rows);
    else if (rows >= 24)
      play_arcade(tty, equation, dim, n_ranks, columns, rows);
  }
} // namespace ryujin
