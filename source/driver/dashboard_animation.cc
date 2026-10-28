//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <driver/dashboard_animation.h>
#include <driver/dashboard_theme.h>
#include <ryujin/base/random.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <optional>
#include <random>
#include <set>
#include <string>

namespace ryujin::animation
{
  namespace
  {
    using Cell = std::pair<int, int>;

    auto random = seeded_mersenne_twister();

    /* Return a random number in [0, n): */
    int pick(std::size_t n)
    {
      return std::uniform_int_distribution<int>(0, int(n) - 1)(random);
    }

    bool chance(double p)
    {
      return std::bernoulli_distribution(p)(random);
    }

    ftxui::Color random_color()
    {
      return dashboard::theme
          .animation[pick(dashboard::theme.animation.size())];
    }

    void paint(Stage &stage,
               Cell c,
               const std::string &glyph,
               ftxui::Color color,
               bool bold = false)
    {
      auto &pixel = stage.pixel(c.first, c.second);
      pixel.character = glyph;
      pixel.foreground_color = color;
      pixel.bold = bold;
    }

    void sparkle(Stage &stage, Cell c)
    {
      static const std::array<std::string, 9> glyphs = {
          "·", "·", "⋅", "+", "*", "⋆", "✦", "✧", "˚"};
      paint(stage, c, glyphs[pick(glyphs.size())], random_color());
      stage.screen.PixelAt(c.first, c.second).dim = chance(0.5);
    }


    /**
     * Sparkles scattered over the free cells that wander off to other
     * free cells.
     */
    class Sparkle final : public Animation
    {
    public:
      void draw(Stage &stage) override
      {
        sparkles_.clear();
        for (int y = 0; y < stage.screen.dimy(); ++y)
          for (int x = 0; x < stage.screen.dimx(); ++x)
            if (stage.free(x, y) && chance(0.02)) {
              sparkles_.emplace_back(x, y);
              sparkle(stage, sparkles_.back());
            }
      }

      void step(Stage &stage) override
      {
        /* Move about one in sixteen sparkles to another free cell: */
        for (std::size_t i = 0; i < (sparkles_.size() + 15) / 16; ++i) {
          auto &sparkle_cell = sparkles_[pick(sparkles_.size())];
          stage.pixel(sparkle_cell.first, sparkle_cell.second);

          for (int attempt = 0; attempt < 16; ++attempt) {
            const Cell c{pick(stage.screen.dimx()), pick(stage.screen.dimy())};
            if (stage.free(c.first, c.second) &&
                std::ranges::find(sparkles_, c) == sparkles_.end()) {
              sparkle_cell = c;
              break;
            }
          }

          sparkle(stage, sparkle_cell);
        }
      }

    private:
      std::vector<Cell> sparkles_;
    };


    /**
     * Snakes that roam the free cells of a periodic stage, hunt for food,
     * and burst into sparkles when they run into a dead end.
     */
    class Snakes final : public Animation
    {
    public:
      void draw(Stage &stage) override
      {
        width_ = stage.screen.dimx();
        height_ = stage.screen.dimy();
        cells_.assign(width_ * height_, empty);

        /*
         * Keep every snake up to the first segment that no longer fits on
         * the stage, or that got separated from its predecessor:
         */
        for (auto &snake : snakes_) {
          auto &body = snake.body;
          std::size_t n = 0;
          for (; n < body.size() && vacant(stage, body[n]) &&
                 (n == 0 || distance(body[n - 1], body[n]) == 1);
               ++n)
            cell(body[n]) = occupied;
          body.resize(n);
          paint_snake(stage, snake);
        }
        std::erase_if(snakes_, [](const Snake &s) { return s.body.empty(); });

        std::erase_if(food_, [&](const Food &food) {
          if (!unused(stage, food.cell))
            return true;
          cell(food.cell) = morsel;
          paint(stage, food.cell, "◦", food.color);
          return false;
        });

        std::erase_if(particles_, [&](const Particle &particle) {
          if (!unused(stage, particle.cell))
            return true;
          sparkle(stage, particle.cell);
          return false;
        });

        /* Populate the free area sparsely: */
        int area = 0;
        for (int y = 0; y < height_; ++y)
          for (int x = 0; x < width_; ++x)
            area += stage.free(x, y);
        max_snakes_ = area < 200 ? 0 : std::clamp(area / 300, 2, 32);
        max_food_ = std::min(area / 150, 64);
      }

      void step(Stage &stage) override
      {
        /* Let the sparkles of explosions twinkle and fade away: */
        std::erase_if(particles_, [&](Particle &particle) {
          const auto [x, y] = particle.cell;
          if (cell(particle.cell) == empty) {
            if (particle.lifetime > 1)
              sparkle(stage, particle.cell);
            else
              stage.pixel(x, y);
          }
          return --particle.lifetime == 0;
        });

        if (snakes_.size() < max_snakes_ && chance(0.1))
          if (const auto start = random_unused_cell(stage)) {
            snakes_.push_back({.body = {*start},
                               .direction = pick(4),
                               .growth = 4,
                               .color = random_color()});
            cell(*start) = occupied;
            paint_snake(stage, snakes_.back());
          }

        if (food_.size() < max_food_ && chance(0.3))
          if (const auto where = random_unused_cell(stage)) {
            food_.push_back({.cell = *where, .color = random_color()});
            cell(*where) = morsel;
            paint(stage, *where, "◦", food_.back().color);
          }

        std::erase_if(snakes_, [&](Snake &snake) {
          if (move(stage, snake))
            return false;
          explode(stage, snake);
          return true;
        });
      }

    private:
      struct Snake {
        /* The segments, starting with the head: */
        std::deque<Cell> body;
        int direction;
        int growth;
        ftxui::Color color;
      };

      struct Food {
        Cell cell;
        ftxui::Color color;
      };

      struct Particle {
        Cell cell;
        int lifetime;
      };

      enum Content : std::uint8_t { empty, occupied, morsel };

      static constexpr std::size_t max_length = 150;

      /* Move the snake by one cell, return false if it is stuck: */
      bool move(Stage &stage, Snake &snake)
      {
        struct Option {
          int direction;
          Cell cell;
          int room;
        };

        /*
         * The options of going straight, left, and right, and how much
         * room each of them leaves within a small horizon:
         */
        const Cell head = snake.body.front();
        const int horizon = int(std::min(snake.body.size(), max_length)) + 4;

        std::vector<Option> options;
        for (const int turn : {0, 3, 1}) {
          const int direction = (snake.direction + turn) % 4;
          const Cell next = neighbor(head, direction);
          if (vacant(stage, next))
            options.push_back({direction, next, room(stage, next, horizon)});
        }
        if (options.empty())
          return false;

        /* Avoid running into dead ends: */
        const int most = std::ranges::max(options, {}, &Option::room).room;
        std::erase_if(options,
                      [most](const Option &o) { return o.room < most; });

        /*
         * Turn at random once in a while, otherwise home in on food in
         * sight, otherwise go straight:
         */
        Option choice = options[0];
        const auto food = nearest_food(head);
        if (chance(0.1) || (!food && choice.direction != snake.direction))
          choice = options[pick(options.size())];
        else if (food)
          choice = std::ranges::min(options, {}, [&](const Option &o) {
            return distance(o.cell, *food);
          });

        const Cell next = choice.cell;
        snake.direction = choice.direction;

        if (cell(next) == morsel) {
          std::erase_if(food_, [&](const Food &f) { return f.cell == next; });
          snake.growth += 3;
        }

        snake.body.push_front(next);
        cell(next) = occupied;

        if (snake.growth > 0 && snake.body.size() <= max_length) {
          --snake.growth;
        } else {
          const auto [x, y] = snake.body.back();
          cell(snake.body.back()) = empty;
          stage.pixel(x, y);
          snake.body.pop_back();
        }

        paint_snake(stage, snake, /*ends*/ true);
        return true;
      }

      /* Paint the snake, or only its head, neck, and tail: */
      void paint_snake(Stage &stage, const Snake &snake, bool ends = false)
      {
        const std::size_t size = snake.body.size();
        for (std::size_t i = 0; i < size; ++i) {
          if (ends && i > 1 && i + 1 < size)
            continue;
          const char *glyph = i == 0 ? "●" : i + 1 == size ? "·" : "•";
          paint(stage, snake.body[i], glyph, snake.color, /*bold*/ i == 0);
        }
      }

      /* Count the vacant cells reachable from @p start, up to @p limit: */
      int room(const Stage &stage, Cell start, int limit)
      {
        std::set<Cell> seen = {start};
        std::vector<Cell> stack = {start};

        int count = 0;
        for (; !stack.empty() && count < limit; ++count) {
          const Cell current = stack.back();
          stack.pop_back();
          for (int direction = 0; direction < 4; ++direction) {
            const Cell next = neighbor(current, direction);
            if (vacant(stage, next) && seen.insert(next).second)
              stack.push_back(next);
          }
        }
        return count;
      }

      /* The nearest food within sight: */
      std::optional<Cell> nearest_food(Cell head) const
      {
        std::optional<Cell> result;
        int best = 25;
        for (const auto &food : food_)
          if (const int d = distance(head, food.cell); d < best) {
            best = d;
            result = food.cell;
          }
        return result;
      }

      std::optional<Cell> random_unused_cell(const Stage &stage)
      {
        for (int attempt = 0; attempt < 16; ++attempt) {
          const Cell c{pick(width_), pick(height_)};
          if (unused(stage, c))
            return c;
        }
        return std::nullopt;
      }

      /* Replace the snake by a small burst of sparkles: */
      void explode(Stage &stage, const Snake &snake)
      {
        const auto burst = [&](Cell c, int lifetime) {
          particles_.push_back({c, lifetime});
          sparkle(stage, c);
        };

        for (const Cell &segment : snake.body) {
          cell(segment) = empty;
          burst(segment, 2 + pick(4));

          const Cell spark =
              wrap({segment.first + pick(3) - 1, segment.second + pick(3) - 1});
          if (chance(0.5) && unused(stage, spark))
            burst(spark, 1 + pick(3));
        }
      }

      /* Whether the cell is on the stage, free, and not part of a snake: */
      bool vacant(const Stage &stage, Cell c)
      {
        return c.first < width_ && c.second < height_ &&
               stage.free(c.first, c.second) && cell(c) != occupied;
      }

      /* Whether the cell is vacant and has no food: */
      bool unused(const Stage &stage, Cell c)
      {
        return vacant(stage, c) && cell(c) == empty;
      }

      Cell wrap(Cell c) const
      {
        return {(c.first + width_) % width_, (c.second + height_) % height_};
      }

      Cell neighbor(Cell c, int direction) const
      {
        static constexpr std::array<Cell, 4> steps = {
            {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}};
        const auto [dx, dy] = steps[direction];
        return wrap({c.first + dx, c.second + dy});
      }

      /* The Manhattan distance on the periodic stage: */
      int distance(Cell a, Cell b) const
      {
        const int dx = std::abs(a.first - b.first);
        const int dy = std::abs(a.second - b.second);
        return std::min(dx, width_ - dx) + std::min(dy, height_ - dy);
      }

      std::uint8_t &cell(Cell c)
      {
        return cells_[c.second * width_ + c.first];
      }

      int width_ = 0;
      int height_ = 0;
      std::size_t max_snakes_ = 0;
      std::size_t max_food_ = 0;

      std::vector<Snake> snakes_;
      std::vector<Food> food_;
      std::vector<Particle> particles_;

      /* The Content of every cell of the stage: */
      std::vector<std::uint8_t> cells_;
    };
  } // namespace


  std::unique_ptr<Animation> create(DashboardAnimation kind)
  {
    switch (kind) {
    case DashboardAnimation::none:
      return nullptr;
    case DashboardAnimation::sparkle:
      return std::make_unique<Sparkle>();
    case DashboardAnimation::matthias:
      return std::make_unique<Snakes>();
    }

    __builtin_unreachable();
  }
} // namespace ryujin::animation
