//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <driver/dashboard_animation.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <optional>
#include <random>
#include <string>
#include <utility>

namespace ryujin::animation
{
  namespace
  {
    const std::vector<std::string> sparkle_glyphs = {
        "·", "·", "⋅", "+", "*", "⋆", "✦", "✧", "˚"};


    /**
     * A source of random numbers for the animations.
     */
    class Dice
    {
    protected:
      /* Return a random index in [0, n): */
      std::size_t pick(std::size_t n)
      {
        return std::uniform_int_distribution<std::size_t>(0, n - 1)(random_);
      }

      bool chance(double p)
      {
        return std::uniform_real_distribution<double>(0., 1.)(random_) < p;
      }

    private:
      std::mt19937 random_;
    };


    /**
     * Sparkles scattered over the free cells that wander off to other
     * free cells.
     */
    class Sparkle final : public Animation, private Dice
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

        /* Move about one in sixteen sparkles to a free cell: */
        const auto n = std::max<std::size_t>(1, sparkles_.size() / 16);
        for (std::size_t i = 0; i < n; ++i) {
          auto &[x, y] = sparkles_[pick(sparkles_.size())];
          stage.pixel(x, y) = ftxui::Pixel();

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
        auto &pixel = stage.pixel(x, y);
        pixel = ftxui::Pixel();
        pixel.character = sparkle_glyphs[pick(sparkle_glyphs.size())];
        pixel.foreground_color = stage.colors[pick(stage.colors.size())];
        pixel.dim = chance(0.5);
      }

      std::vector<std::pair<int, int>> sparkles_;
    };


    /**
     * Snakes that roam the free cells of a periodic stage, hunt for food,
     * and burst into sparkles when they run into a dead end.
     */
    class Snakes final : public Animation, private Dice
    {
    public:
      void draw(Stage &stage, const DashboardData & /*data*/) override
      {
        width_ = stage.screen.dimx();
        height_ = stage.screen.dimy();
        cells_.assign(std::size_t(width_) * height_, empty);
        visited_.assign(cells_.size(), 0);
        generation_ = 0;

        /*
         * Keep every snake up to the first segment that no longer fits on
         * the stage, or that got separated from its predecessor:
         */
        for (auto &snake : snakes_) {
          std::size_t n = 0;
          for (; n < snake.body.size(); ++n) {
            const Cell segment = snake.body[n];
            if (!vacant(stage, segment) ||
                (n > 0 && distance(snake.body[n - 1], segment) != 1))
              break;
            cell(segment) = occupied;
          }
          snake.body.resize(n);
        }
        std::erase_if(snakes_,
                      [](const Snake &snake) { return snake.body.empty(); });

        std::erase_if(food_, [&](const Food &food) {
          return !vacant(stage, food.cell) || cell(food.cell) != empty;
        });
        for (const auto &food : food_)
          cell(food.cell) = morsel;

        std::erase_if(particles_, [&](const Particle &particle) {
          return !vacant(stage, particle.cell) || cell(particle.cell) != empty;
        });

        /* Populate the free area sparsely: */
        int area = 0;
        for (int y = 0; y < height_; ++y)
          for (int x = 0; x < width_; ++x)
            area += stage.free(x, y);
        max_snakes_ = area < 200 ? 0 : std::clamp(area / 300, 2, 32);
        max_food_ = std::min(area / 150, 64);

        for (const auto &snake : snakes_)
          paint_snake(stage, snake);
        for (const auto &food : food_)
          paint(stage, food.cell, "◦", food.color);
        for (const auto &particle : particles_)
          sparkle(stage, particle.cell);
      }

      void step(Stage &stage, const DashboardData & /*data*/) override
      {
        fade(stage);

        if (snakes_.size() < max_snakes_ && chance(0.1))
          spawn_snake(stage);
        if (food_.size() < max_food_ && chance(0.3))
          spawn_food(stage);

        for (std::size_t i = 0; i < snakes_.size();) {
          if (move(stage, snakes_[i])) {
            ++i;
          } else {
            explode(stage, snakes_[i]);
            snakes_.erase(snakes_.begin() + i);
          }
        }
      }

    private:
      using Cell = std::pair<int, int>;

      struct Snake {
        /* The segments, starting with the head: */
        std::deque<Cell> body;
        int direction;
        int growth;
        std::size_t color;
      };

      struct Food {
        Cell cell;
        std::size_t color;
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
         * Collect the options of going straight, left, and right, and
         * how much room each of them leaves within a small horizon:
         */
        const Cell head = snake.body.front();
        const int horizon =
            static_cast<int>(std::min(snake.body.size(), max_length)) + 4;

        std::array<Option, 3> options;
        std::size_t n = 0;
        int most_room = 0;
        for (const int turn : {0, 3, 1}) {
          const int direction = (snake.direction + turn) % 4;
          const Cell next = neighbor(head, direction);
          if (!vacant(stage, next))
            continue;
          options[n] = {.direction = direction,
                        .cell = next,
                        .room = room(stage, next, horizon)};
          most_room = std::max(most_room, options[n].room);
          ++n;
        }

        if (n == 0)
          return false;

        /* Avoid running into dead ends: */
        const auto tight = [&](const Option &o) { return o.room < most_room; };
        n = std::remove_if(options.begin(), options.begin() + n, tight) -
            options.begin();

        /*
         * Turn at random once in a while, otherwise home in on food in
         * sight, otherwise go straight:
         */
        const Option *choice = &options[0];
        const auto food = nearest_food(head);
        if (chance(0.1) || (!food && choice->direction != snake.direction)) {
          choice = &options[pick(n)];
        } else if (food) {
          choice = std::min_element(options.begin(),
                                    options.begin() + n,
                                    [&](const Option &a, const Option &b) {
                                      return distance(a.cell, *food) <
                                             distance(b.cell, *food);
                                    });
        }

        const Cell next = choice->cell;
        snake.direction = choice->direction;

        if (cell(next) == morsel) {
          std::erase_if(food_,
                        [&](const Food &food) { return food.cell == next; });
          snake.growth += 3;
        }

        snake.body.push_front(next);
        cell(next) = occupied;

        if (snake.growth > 0 && snake.body.size() <= max_length) {
          --snake.growth;
        } else {
          const Cell tail = snake.body.back();
          snake.body.pop_back();
          cell(tail) = empty;
          stage.pixel(tail.first, tail.second) = ftxui::Pixel();
        }

        paint_snake(stage, snake, /*ends*/ true);
        return true;
      }

      /* Paint the snake, or only its head, neck, and tail: */
      void paint_snake(Stage &stage, const Snake &snake, bool ends = false)
      {
        const auto &body = snake.body;
        const std::size_t size = body.size();
        for (std::size_t i = 0; i < size; ++i) {
          if (ends && i > 1 && i + 1 < size)
            continue;
          const char *glyph = i == 0 ? "●" : i + 1 == size ? "·" : "•";
          paint(stage, body[i], glyph, snake.color, /*bold*/ i == 0);
        }
      }

      /* Count the vacant cells reachable from @p start, up to @p limit: */
      int room(const Stage &stage, Cell start, int limit)
      {
        if (++generation_ == 0) {
          std::ranges::fill(visited_, 0);
          generation_ = 1;
        }

        stack_.clear();
        stack_.push_back(start);
        visited(start) = generation_;

        int count = 0;
        while (!stack_.empty() && count < limit) {
          const Cell current = stack_.back();
          stack_.pop_back();
          ++count;
          for (int direction = 0; direction < 4; ++direction) {
            const Cell next = neighbor(current, direction);
            if (vacant(stage, next) && visited(next) != generation_) {
              visited(next) = generation_;
              stack_.push_back(next);
            }
          }
        }
        return count;
      }

      std::optional<Cell> nearest_food(Cell head) const
      {
        constexpr int sight = 24;

        std::optional<Cell> result;
        int best = sight + 1;
        for (const auto &food : food_)
          if (const int d = distance(head, food.cell); d < best) {
            best = d;
            result = food.cell;
          }
        return result;
      }

      void spawn_snake(Stage &stage)
      {
        if (const auto start = random_empty_cell(stage)) {
          snakes_.push_back(Snake{.body = {*start},
                                  .direction = static_cast<int>(pick(4)),
                                  .growth = 4,
                                  .color = pick(stage.colors.size())});
          cell(*start) = occupied;
          paint_snake(stage, snakes_.back());
        }
      }

      void spawn_food(Stage &stage)
      {
        if (const auto where = random_empty_cell(stage)) {
          food_.push_back(
              Food{.cell = *where, .color = pick(stage.colors.size())});
          cell(*where) = morsel;
          paint(stage, *where, "◦", food_.back().color);
        }
      }

      std::optional<Cell> random_empty_cell(const Stage &stage)
      {
        for (int attempt = 0; attempt < 16; ++attempt) {
          const Cell candidate{static_cast<int>(pick(width_)),
                               static_cast<int>(pick(height_))};
          if (stage.free(candidate.first, candidate.second) &&
              cell(candidate) == empty)
            return candidate;
        }
        return std::nullopt;
      }

      /* Replace the snake by a small burst of sparkles: */
      void explode(Stage &stage, const Snake &snake)
      {
        for (const Cell &segment : snake.body) {
          cell(segment) = empty;
          particles_.push_back(
              {.cell = segment, .lifetime = 2 + static_cast<int>(pick(4))});
          sparkle(stage, segment);

          const Cell spark{segment.first + static_cast<int>(pick(3)) - 1,
                           segment.second + static_cast<int>(pick(3)) - 1};
          if (chance(0.5) && vacant(stage, wrap(spark)) &&
              cell(wrap(spark)) == empty) {
            particles_.push_back({.cell = wrap(spark),
                                  .lifetime = 1 + static_cast<int>(pick(3))});
            sparkle(stage, wrap(spark));
          }
        }
      }

      /* Let the sparkles of explosions twinkle and fade away: */
      void fade(Stage &stage)
      {
        std::erase_if(particles_, [&](Particle &particle) {
          const bool visible = cell(particle.cell) == empty;
          if (--particle.lifetime > 0) {
            if (visible)
              sparkle(stage, particle.cell);
            return false;
          }
          if (visible)
            stage.pixel(particle.cell.first, particle.cell.second) =
                ftxui::Pixel();
          return true;
        });
      }

      void sparkle(Stage &stage, Cell c)
      {
        paint(stage,
              c,
              sparkle_glyphs[pick(sparkle_glyphs.size())],
              pick(stage.colors.size()));
      }

      static void paint(Stage &stage,
                        Cell c,
                        const std::string &glyph,
                        std::size_t color,
                        bool bold = false)
      {
        auto &pixel = stage.pixel(c.first, c.second);
        pixel = ftxui::Pixel();
        pixel.character = glyph;
        pixel.foreground_color = stage.colors[color % stage.colors.size()];
        pixel.bold = bold;
      }

      /* Whether the cell is on the stage, free, and not part of a snake: */
      bool vacant(const Stage &stage, Cell c) const
      {
        return c.first < width_ && c.second < height_ &&
               stage.free(c.first, c.second) && cell(c) != occupied;
      }

      Cell wrap(Cell c) const
      {
        return {(c.first + width_) % width_, (c.second + height_) % height_};
      }

      Cell neighbor(Cell c, int direction) const
      {
        static constexpr std::array<Cell, 4> steps = {
            {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}};
        return wrap({c.first + steps[direction].first,
                     c.second + steps[direction].second});
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
        return cells_[static_cast<std::size_t>(c.second) * width_ + c.first];
      }

      std::uint8_t cell(Cell c) const
      {
        return cells_[static_cast<std::size_t>(c.second) * width_ + c.first];
      }

      std::uint32_t &visited(Cell c)
      {
        return visited_[static_cast<std::size_t>(c.second) * width_ + c.first];
      }

      int width_ = 0;
      int height_ = 0;
      std::size_t max_snakes_ = 0;
      std::size_t max_food_ = 0;

      std::vector<Snake> snakes_;
      std::vector<Food> food_;
      std::vector<Particle> particles_;

      /* The contents of every cell of the stage: */
      std::vector<std::uint8_t> cells_;

      /* Scratch space for room(): */
      std::vector<std::uint32_t> visited_;
      std::uint32_t generation_ = 0;
      std::vector<Cell> stack_;
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
