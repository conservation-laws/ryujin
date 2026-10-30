//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <driver/dashboard_animation.h>
#include <driver/dashboard_theme.h>
#include <ryujin/base/random.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <initializer_list>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <string_view>

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
               bool bold = false,
               bool dim = false)
    {
      auto &pixel = stage.pixel(c.first, c.second);
      pixel.character = glyph;
      pixel.foreground_color = color;
      pixel.bold = bold;
      pixel.dim = dim;
    }

    bool open(const Stage &stage, int x, int y)
    {
      return x >= 0 && y >= 0 && x < stage.screen.dimx() &&
             y < stage.screen.dimy() && stage.free(x, y);
    }

    ftxui::Color shade(double t)
    {
      const auto &colors = dashboard::theme.animation;
      const int n = static_cast<int>(colors.size());
      return colors[std::clamp(static_cast<int>(t * n), 0, n - 1)];
    }

    int free_area(const Stage &stage)
    {
      int area = 0;
      for (int y = 0; y < stage.screen.dimy(); ++y)
        for (int x = 0; x < stage.screen.dimx(); ++x)
          area += stage.free(x, y);
      return area;
    }

    using Sprite = std::vector<std::vector<std::string>>;

    Sprite make_sprite(std::initializer_list<std::string_view> rows)
    {
      Sprite sprite;
      for (const auto row : rows) {
        auto &glyphs = sprite.emplace_back();
        for (const char c : row)
          if ((c & 0xc0) == 0x80)
            glyphs.back() += c;
          else
            glyphs.emplace_back(1, c);
      }
      return sprite;
    }

    Sprite mirror(Sprite sprite)
    {
      static const std::map<std::string, std::string> swap = {{"(", ")"},
                                                              {")", "("},
                                                              {"/", "\\"},
                                                              {"\\", "/"},
                                                              {"<", ">"},
                                                              {">", "<"}};
      for (auto &row : sprite) {
        std::ranges::reverse(row);
        for (auto &glyph : row)
          if (const auto it = swap.find(glyph); it != swap.end())
            glyph = it->second;
      }
      return sprite;
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


    /**
     * Fish, crabs, and jellyfish that swim through the free cells.
     */
    class Aquarium final : public Animation
    {
    public:
      void draw(Stage &stage) override
      {
        width_ = stage.screen.dimx();
        height_ = stage.screen.dimy();

        if (static_cast<int>(weeds_.size()) != std::max(1, width_ / 8)) {
          weeds_.clear();
          const int n = std::max(1, width_ / 8);
          for (int k = 0; k < n; ++k)
            weeds_.push_back(
                {static_cast<int>(width_ * (k + 0.2 + 0.6 * pick(100) / 100.) /
                                  n),
                 3 + pick(7),
                 pick(2),
                 shade(0.2 + 0.5 * pick(100) / 100.)});
        }

        std::erase_if(fish_, [&](const Fish &f) {
          return !fits(stage, look(f), f.x, f.y);
        });
        const auto count = [&](auto... kinds) {
          return std::ranges::count_if(
              fish_, [&](const Fish &f) { return ((f.kind == kinds) || ...); });
        };
        for (auto n = count(Kind::small, Kind::big);
             n < std::clamp(free_area(stage) / 250, 1, 12);
             ++n)
          spawn(stage, chance(0.4) ? Kind::big : Kind::small);
        for (const Kind kind : {Kind::puffer, Kind::crab, Kind::jelly})
          if (count(kind) == 0)
            spawn(stage, kind);

        bubbles_.clear();
        scene_.clear();
        compose(stage);
      }

      void step(Stage &stage) override
      {
        ++time_;
        for (auto &f : fish_)
          swim(stage, f);

        for (const auto &w : weeds_)
          if (chance(0.03))
            bubble(w.x, height_ - 2 - w.height);

        std::erase_if(bubbles_, [&](Bubble &b) {
          b.y -= 0.5;
          b.x += (pick(3) - 1) * 0.3;
          ++b.age;
          return !open(stage, round(b.x), round(b.y));
        });

        compose(stage);
      }

    private:
      enum class Kind : std::uint8_t { small, big, puffer, crab, jelly };

      struct Fish {
        Kind kind;
        double x;
        int y;
        int dir;
        int timer;
        bool alt;
        ftxui::Color color;
      };

      struct Weed {
        int x;
        int height;
        int phase;
        ftxui::Color color;
      };

      struct Bubble {
        double x;
        double y;
        int age;
      };

      struct Glyph {
        std::string text;
        ftxui::Color color;
        bool bold = false;
        bool dim = false;
        bool operator==(const Glyph &) const = default;
      };

      const std::vector<Sprite> right_ = {
          make_sprite({"><>"}),
          make_sprite({"><(((°>"}),
          make_sprite({">(°)"}),
          make_sprite({"  \\ | /  ", ">-(  ° )-", "  / | \\  "}),
          make_sprite({"V(°.°)V"}),
          make_sprite({"Y(°.°)Y"}),
          make_sprite({" .--. ", "(    )", " )()( ", " (  ) ", " )  ( "}),
          make_sprite({" .--. ", "(    )", " ()() ", " )  ( ", " (  ) "}),
          make_sprite({" .-. ", "(   )", " ||| ", " | | ", " | | "})};
      const std::vector<Sprite> left_ = [this] {
        std::vector<Sprite> sprites;
        for (const auto &sprite : right_)
          sprites.push_back(mirror(sprite));
        return sprites;
      }();

      static int round(double v)
      {
        return static_cast<int>(std::lround(v));
      }

      const Sprite &look(const Fish &f) const
      {
        const auto frame = [&] {
          switch (f.kind) {
          case Kind::small:
            return 0;
          case Kind::big:
            return 1;
          case Kind::puffer:
            return f.alt ? 3 : 2;
          case Kind::crab:
            return f.alt && f.timer / 3 % 2 ? 5 : 4;
          case Kind::jelly:
            return f.alt ? 8 : 6 + time_ / 4 % 2;
          }
          __builtin_unreachable();
        }();
        return (f.dir > 0 ? right_ : left_)[frame];
      }

      bool fits(const Stage &stage, const Sprite &sprite, double x, int y) const
      {
        if (y + static_cast<int>(sprite.size()) > height_ - 1)
          return false;
        for (int r = 0; r < static_cast<int>(sprite.size()); ++r)
          for (int k = 0; k < static_cast<int>(sprite[r].size()); ++k)
            if (sprite[r][k] != " " && !open(stage, round(x) + k, y + r))
              return false;
        return true;
      }

      bool fits(const Stage &stage, const Fish &f) const
      {
        return fits(stage, look(f), f.x, f.y);
      }

      void spawn(const Stage &stage, Kind kind)
      {
        for (int attempt = 0; attempt < 100; ++attempt) {
          const Fish f{kind,
                       static_cast<double>(pick(width_)),
                       kind == Kind::crab ? height_ - 2 : pick(height_),
                       chance(0.5) ? 1 : -1,
                       60 + pick(80),
                       false,
                       random_color()};
          if (fits(stage, f)) {
            fish_.push_back(f);
            return;
          }
        }
      }

      void bubble(double x, int y)
      {
        bubbles_.push_back({x, static_cast<double>(y), 0});
      }

      void advance(const Stage &stage, Fish &f, double speed)
      {
        if (const double x = f.x + f.dir * speed;
            fits(stage, look(f), x, f.y) && !chance(0.003))
          f.x = x;
        else
          f.dir = -f.dir;
      }

      bool change(const Stage &stage, Fish &f, bool alt, int dy)
      {
        Fish g = f;
        g.alt = alt;
        g.y += dy;
        if (!fits(stage, g))
          return false;
        f = g;
        return true;
      }

      void swim(const Stage &stage, Fish &f)
      {
        switch (f.kind) {
        case Kind::small:
        case Kind::big:
          if (const int dy = chance(0.5) ? 1 : -1; chance(0.02))
            change(stage, f, false, dy);
          advance(stage, f, f.kind == Kind::big ? 0.25 : 0.45);
          if (chance(0.01))
            bubble(f.x + (f.dir > 0 ? look(f)[0].size() : -1.), f.y);
          return;

        case Kind::puffer:
          if (--f.timer <= 0) {
            if (change(stage, f, !f.alt, f.alt ? 1 : -1)) {
              f.timer = f.alt ? 25 + pick(15) : 60 + pick(80);
              for (int k = 0; f.alt && k < 3; ++k)
                bubble(f.x + pick(6), f.y - 1);
            } else
              f.timer = 10;
          }
          if (!f.alt)
            advance(stage, f, 0.15);
          return;

        case Kind::crab:
          if (f.alt)
            f.alt = --f.timer > 0;
          else if (chance(0.01)) {
            f.alt = true;
            f.timer = 20 + pick(20);
          } else
            advance(stage, f, 0.2);
          return;

        case Kind::jelly:
          if (f.alt) {
            if (f.timer > 5)
              change(stage, f, true, -1);
            if (--f.timer <= 0 && change(stage, f, false, 0))
              f.timer = 14 + pick(13);
          } else {
            if (f.timer % 4 == 0 && !change(stage, f, false, 1))
              f.timer = 1;
            advance(stage, f, 0.05);
            if (--f.timer <= 0 && change(stage, f, true, 0))
              f.timer = 7 + pick(7);
          }
          return;
        }
      }

      void compose(Stage &stage)
      {
        static constexpr std::array<const char *, 7> sand = {
            "_", ".", "_", "_", ",", "_", "."};
        std::map<int, Glyph> next;
        const auto put = [&](int x, int y, Glyph g) {
          if (open(stage, x, y))
            next[y * width_ + x] = std::move(g);
        };

        const auto floor = dashboard::theme.animation.front();
        for (int x = 0; x < width_; ++x)
          put(x, height_ - 1, {sand[x * 5 % sand.size()], floor, false, true});

        for (const auto &w : weeds_)
          for (int r = 0; r < w.height && open(stage, w.x, height_ - 2 - r);
               ++r)
            put(w.x,
                height_ - 2 - r,
                {(time_ / 8 + r + w.phase) % 2 ? "(" : ")", w.color});

        const auto bright = dashboard::theme.animation.back();
        for (const auto &b : bubbles_)
          put(round(b.x),
              round(b.y),
              {b.age < 6    ? "."
               : b.age < 16 ? "o"
                            : "O",
               bright,
               false,
               b.age < 6});

        for (const bool front : {false, true})
          for (const auto &f : fish_) {
            if ((f.kind == Kind::jelly || f.kind == Kind::crab) == front)
              continue;
            const auto &sprite = look(f);
            for (int r = 0; r < static_cast<int>(sprite.size()); ++r)
              for (int k = 0; k < static_cast<int>(sprite[r].size()); ++k)
                if (sprite[r][k] != " ")
                  put(round(f.x) + k,
                      f.y + r,
                      {sprite[r][k], f.color, true, f.kind == Kind::jelly});
          }

        for (const auto &[i, g] : scene_)
          if (!next.contains(i))
            stage.pixel(i % width_, i / width_);
        for (const auto &[i, g] : next)
          if (const auto it = scene_.find(i);
              it == scene_.end() || !(it->second == g))
            paint(stage,
                  {i % width_, i / width_},
                  g.text,
                  g.color,
                  g.bold,
                  g.dim);
        scene_ = std::move(next);
      }

      int width_ = 0;
      int height_ = 0;
      int time_ = 0;
      std::vector<Fish> fish_;
      std::vector<Weed> weeds_;
      std::vector<Bubble> bubbles_;
      std::map<int, Glyph> scene_;
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
    case DashboardAnimation::wyatt:
      return std::make_unique<Aquarium>();
    }

    __builtin_unreachable();
  }
} // namespace ryujin::animation
