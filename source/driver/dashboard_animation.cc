//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <driver/dashboard_animation.h>
#include <driver/dashboard_theme.h>
#include <ryujin/base/random.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
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
               bool bold = false)
    {
      auto &pixel = stage.pixel(c.first, c.second);
      pixel.character = glyph;
      pixel.foreground_color = color;
      pixel.bold = bold;
    }

    /** The content of a cell of a frame, empty if the cell is blank. */
    struct Glyph {
      std::string character;
      ftxui::Color color;
      bool bold = false;
      bool dim = false;

      bool operator==(const Glyph &) const = default;
    };

    /*
     * Paint the cells of the @p frame of a region of the stage, row by
     * row, that differ from the @p last frame, and make @p frame the last
     * frame:
     */
    void repaint(Stage &stage,
                 const ftxui::Box &region,
                 std::vector<Glyph> &last,
                 std::vector<Glyph> frame)
    {
      const int width = region.x_max - region.x_min + 1;
      if (last.size() != frame.size())
        last.assign(frame.size(), Glyph());

      for (std::size_t i = 0; i < frame.size(); ++i) {
        const auto &glyph = frame[i];
        if (glyph == last[i])
          continue;
        auto &pixel = stage.pixel(region.x_min + int(i) % width,
                                  region.y_min + int(i) / width);
        if (!glyph.character.empty()) {
          pixel.character = glyph.character;
          pixel.foreground_color = glyph.color;
          pixel.bold = glyph.bold;
          pixel.dim = glyph.dim;
        }
      }

      last = std::move(frame);
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
     * A pour-over coffee bar below the dashboard: a barista walks up,
     * weighs and grinds the beans, and brews a pot of coffee that is
     * ready when the simulation finishes, or after about two and a half
     * minutes if the progress of the simulation is unknown.
     */
    class Coffee final : public Animation
    {
    public:
      void draw(Stage &stage) override
      {
        place(stage);
        sync(stage);
        frame_.clear();
        show(stage);
      }

      void step(Stage &stage) override
      {
        tick(stage);
        sync(stage);
        show(stage);
      }

      /* Serve the coffee when the simulation finishes: */
      void finish(Stage &stage) override
      {
        place(stage);
        phase_ = Phase::done;
        barista_ = station();
        brew_ = 1.;
        level_ = 1.;
        water_ = bloom_water;
        for (const auto &pour : pours)
          water_ += pour.water;
        frame_.clear();
        show(stage);
      }

    private:
      enum class Phase : std::uint8_t { weigh, grind, bloom, pour, done };

      /* The duration of the phases in steps of the animation: */
      static constexpr int weigh_steps = 25;
      static constexpr int grind_steps = 30;
      static constexpr int bloom_steps = 200;
      static constexpr int pour_steps = 800;

      /* The water of the bloom in grams, poured in the first steps: */
      static constexpr double bloom_water = 75.;
      static constexpr int bloom_pour_steps = 30;

      /*
       * The pours after the bloom: the water in grams, and the fractions
       * of the brew at which the pour begins and ends:
       */
      struct Pour {
        double water;
        double begin;
        double end;
      };
      static constexpr std::array<Pour, 3> pours = {
          {{75., 0., 0.15}, {50., 0.33, 0.43}, {100., 0.6, 0.8}}};

      /* The size of the coffee bar: */
      static constexpr int bar_width = 58;
      static constexpr int bar_height = 10;

      /* The columns of the stations, relative to the left end: */
      static constexpr int bag_x = 0;
      static constexpr int scale_x = 15;
      static constexpr int grinder_x = 27;
      static constexpr int dripper_x = 38;
      static constexpr int kettle_x = 46;

      /* The column of the barista before walking up to the bar: */
      static constexpr int barista_start = -6;

      /* The duration of a step of the animation: */
      static constexpr std::chrono::milliseconds step_duration{150};

      /*
       * The steps of the preparations before the pour: walking to the
       * stations at a column every two steps, weighing, grinding, and the
       * bloom:
       */
      static constexpr double preparation_steps =
          2 * (dripper_x - 5 - barista_start) + weigh_steps + grind_steps +
          bloom_steps;

      /* The time on the brew timer at the end of the bloom and in total: */
      static constexpr int bloom_seconds = 30;
      static constexpr int brew_seconds = 150;

      static std::string_view pick_roaster()
      {
        static constexpr std::array<std::string_view, 4> others = {
            "Corvus", "Native", "Freeform", "Iconik"};
        return chance(0.6) ? "Jitterz" : others[pick(others.size())];
      }

      static ftxui::Color coffee()
      {
        return ftxui::Color::RGB(150, 95, 55);
      }

      /*
       * Place the bottom left corner of the bar in a free band below,
       * above, left, or right of the dashboard:
       */
      void place(Stage &stage)
      {
        const auto &box = stage.dashboard;
        const int width = stage.screen.dimx();
        const int height = stage.screen.dimy();

        struct Band {
          int x_min, x_max, y_min, y_max;
        };
        const std::array<Band, 4> bands = {{
            {0, width - 1, box.y_max + 1, height - 1},
            {0, width - 1, 0, box.y_min - 1},
            {0, box.x_min - 1, 0, height - 1},
            {box.x_max + 1, width - 1, 0, height - 1},
        }};

        origin_.reset();
        for (const auto &band : bands) {
          const int columns = band.x_max - band.x_min + 1;
          const int rows = band.y_max - band.y_min + 1;
          if (columns >= bar_width && rows >= bar_height) {
            origin_ = Cell{band.x_min + (columns - bar_width) / 2, band.y_max};
            return;
          }
        }

        /*
         * The bar does not fit next to the dashboard. Move the dashboard
         * to the left, or to the top, if that makes room:
         */
        const auto [columns, rows] = stage.free_space();
        if (box.x_min > box.x_max)
          return;
        if (columns >= bar_width && box.x_min > 0)
          stage.shift = {-box.x_min, 0};
        else if (rows >= bar_height && box.y_min > 0)
          stage.shift = {0, -box.y_min};
      }

      /*
       * The column the barista works at in the current phase, to the left
       * of the station, or to the right of the kettle with a cup:
       */
      int station() const
      {
        switch (phase_) {
        case Phase::weigh:
          return scale_x - 5;
        case Phase::grind:
          return grinder_x - 5;
        case Phase::done:
          return kettle_x + 5;
        default:
          return dripper_x - 5;
        }
      }

      /*
       * Advance the animation to the wall time. The dashboard only
       * redraws the whole stage once per update if it does not fit into
       * the terminal, and never calls step() then:
       */
      void sync(const Stage &stage)
      {
        const auto now = std::chrono::steady_clock::now();
        if (!start_)
          start_ = now;
        const auto target = (now - *start_) / step_duration;
        while (ticks_ < target)
          tick(stage);
      }

      /*
       * Advance the animation by one step of wall time. If the simulation
       * finishes soon, hurry through the preparations so that they take
       * no more than about a quarter of the remaining time:
       */
      void tick(const Stage &stage)
      {
        ++ticks_;
        int steps = 1;
        if (phase_ < Phase::pour && stage.eta > 0.) {
          const double budget = 0.25 * stage.eta / 0.15;
          steps = std::clamp(int(std::ceil(preparation_steps / budget)), 1, 8);
        }
        for (int i = 0; i < steps; ++i)
          advance(stage);
      }

      void advance(const Stage &stage)
      {
        ++step_;

        /* Walk over to the station of the current phase first: */
        if (barista_ != station()) {
          if (step_ % 2 == 0)
            barista_ += barista_ < station() ? 1 : -1;
          return;
        }

        const auto enter = [this](Phase phase) {
          phase_ = phase;
          phase_step_ = 0;
        };

        switch (++phase_step_; phase_) {
        case Phase::weigh:
          if (phase_step_ == weigh_steps)
            enter(Phase::grind);
          break;
        case Phase::grind:
          if (phase_step_ == grind_steps)
            enter(Phase::bloom);
          break;
        case Phase::bloom:
          level_ = 0.05 * phase_step_ / bloom_steps;
          water_ = bloom_water * std::min(phase_step_, bloom_pour_steps) /
                   bloom_pour_steps;
          if (phase_step_ == bloom_steps)
            enter(Phase::pour);
          break;
        case Phase::pour: {
          /*
           * Follow the progress of the simulation, or the clock, but
           * brew no faster than in about thirty seconds:
           */
          const double progress = stage.progress >= 0.
                                      ? stage.progress
                                      : double(phase_step_) / pour_steps;
          brew_ += std::clamp(std::min(progress, 1.) - brew_, 0., 1. / 200.);
          if (brew_ >= 1.) {
            brew_ = 1.;
            enter(Phase::done);
          }
          level_ = 0.05 + 0.95 * brew_;
          water_ = bloom_water;
          for (const auto &pour : pours)
            water_ += pour.water *
                      std::clamp((brew_ - pour.begin) / (pour.end - pour.begin),
                                 0.,
                                 1.);
          break;
        }
        case Phase::done:
          break;
        }
      }

      /* Whether water streams from the kettle: */
      bool pouring() const
      {
        if (barista_ != dripper_x - 5)
          return false;
        if (phase_ == Phase::bloom)
          return phase_step_ < bloom_pour_steps;
        return phase_ == Phase::pour &&
               std::ranges::any_of(pours, [this](const Pour &pour) {
                 return pour.begin <= brew_ && brew_ < pour.end;
               });
      }

      /* Compose the current frame and paint the cells that changed: */
      void show(Stage &stage)
      {
        if (!origin_)
          return;

        const int width = stage.screen.dimx();
        const int height = stage.screen.dimy();
        const ftxui::Box region = {origin_->first,
                                   origin_->first + bar_width - 1,
                                   origin_->second - bar_height + 1,
                                   origin_->second};
        std::vector<Glyph> frame(bar_width * bar_height);

        /*
         * Write the text @p row rows above the bottom of the bar,
         * skipping blanks:
         */
        const auto text = [&](int column,
                              int row,
                              std::string_view text,
                              ftxui::Color color,
                              bool bold = false,
                              bool dim = false) {
          int x = origin_->first + column;
          const int y = origin_->second - row;
          for (std::size_t i = 0; i < text.size(); ++x) {
            /* The length of the UTF-8 sequence: */
            const auto lead = static_cast<unsigned char>(text[i]);
            const std::size_t n = lead < 0x80   ? 1
                                  : lead < 0xe0 ? 2
                                  : lead < 0xf0 ? 3
                                                : 4;
            if (text[i] != ' ' && region.Contain(x, y) && x < width &&
                y < height && stage.free(x, y))
              frame[(y - region.y_min) * bar_width + x - region.x_min] = {
                  std::string(text.substr(i, n)), color, bold, dim};
            i += n;
          }
        };

        const auto &theme = dashboard::theme;
        const auto plain = ftxui::Color::Default;
        const bool grinding = phase_ == Phase::grind && barista_ == station();

        /* The counter: */
        for (int x = 0; x < bar_width; ++x)
          text(x, 2, "━", theme.muted);
        for (const int x : {0, bar_width - 1})
          for (const int row : {0, 1})
            text(x, row, "┃", theme.muted);

        /* The bag of beans: */
        text(bag_x, 4, "┌~~~~~~~~┐", theme.muted);
        text(bag_x, 3, "│", theme.muted);
        text(bag_x + 1 + (8 - int(roaster_.size()) + 1) / 2,
             3,
             roaster_,
             coffee(),
             true);
        text(bag_x + 9, 3, "│", theme.muted);

        /* The scale, with the beans in the bowl until they are ground: */
        const double grams = phase_ == Phase::weigh
                                 ? 18. * phase_step_ / weigh_steps
                             : phase_ == Phase::grind && !grinding ? 18.
                                                                   : 0.;
        char display[16];
        std::snprintf(display, sizeof(display), "%4.1fg", grams);
        text(scale_x, 3, "[", theme.muted);
        text(scale_x + 1, 3, display, theme.good, true);
        text(scale_x + 6, 3, "]", theme.muted);
        text(scale_x + 1, 4, "\\___/", theme.muted);
        const int beans = int(std::ceil(3. * grams / 18.));
        text(scale_x + 2, 4, std::string(beans, 'o'), coffee(), true);
        if (phase_ == Phase::weigh && barista_ == station() &&
            phase_step_ < weigh_steps - 4)
          text(scale_x + 3, 5 + step_ % 2, "∘", coffee(), true);

        /* The grinder, shaking while it grinds: */
        const int shake = grinding ? step_ % 2 : 0;
        const bool hopper = phase_ == Phase::weigh ||
                            (grinding && phase_step_ < 2 * grind_steps / 3) ||
                            (phase_ == Phase::grind && !grinding);
        text(grinder_x + 1 + shake, 6, "▄▄▄", theme.muted);
        text(grinder_x + 1 + shake, 5, "\\ /", theme.muted);
        if (hopper)
          text(grinder_x + 2 + shake,
               5,
               grinding && step_ % 2 ? "░" : "▒",
               coffee());
        text(grinder_x + 1 + shake, 4, "|▓|", theme.muted);
        text(grinder_x, 3, "[___]", theme.muted);
        if (grinding)
          text(grinder_x - 1 + 2 * shake, 7, "brrr", theme.muted, false, true);

        /* The dripper on top of the pot: */
        text(dripper_x, 6, "\\     /", theme.muted);
        text(dripper_x + 1, 5, "\\___/", theme.muted);
        text(dripper_x, 4, "(     )", theme.muted);
        text(dripper_x, 3, "(     )", theme.muted);
        if (phase_ >= Phase::bloom) {
          const bool wet = pouring() || phase_ == Phase::pour;
          text(dripper_x + 1,
               6,
               wet && step_ % 4 < 2 ? "~~~~~" : "▒▒▒▒▒",
               wet ? theme.animation[0] : coffee());
        }

        /* The coffee in the pot, in eighths of a cell: */
        static const std::array<std::string, 9> blocks = {
            " ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
        const int eighths = int(std::lround(16. * level_));
        if (eighths > 0)
          text(
              dripper_x + 1,
              3,
              [&] {
                std::string row;
                for (int i = 0; i < 5; ++i)
                  row += blocks[std::min(eighths, 8)];
                return row;
              }(),
              coffee());
        if (eighths > 8)
          text(
              dripper_x + 1,
              4,
              [&] {
                std::string row;
                for (int i = 0; i < 5; ++i)
                  row += blocks[eighths - 8];
                return row;
              }(),
              coffee());
        if (phase_ >= Phase::bloom && phase_ != Phase::done && eighths < 16 &&
            step_ % 3 == 0)
          text(dripper_x + 3, 4, "·", coffee(), true);

        /*
         * The timer of the brew, from the start of the bloom, and the
         * water poured so far:
         */
        if (phase_ >= Phase::bloom) {
          /*
           * The timer runs as if the brew took two and a half minutes,
           * however long the simulation takes:
           */
          const int seconds =
              phase_ == Phase::bloom
                  ? bloom_seconds * phase_step_ / bloom_steps
                  : bloom_seconds +
                        int(std::lround((brew_seconds - bloom_seconds) *
                                        brew_));
          char timer[16];
          std::snprintf(
              timer, sizeof(timer), "%d:%02d", seconds / 60, seconds % 60);
          const int length = int(std::strlen(timer));
          char water[16];
          std::snprintf(water, sizeof(water), "%3.0fg", water_);
          text(kettle_x, 6, timer, theme.good, true);
          text(kettle_x + length + 1, 6, water, theme.good, true);
        }

        /* The kettle, held over the dripper while pouring: */
        if (pouring()) {
          text(dripper_x, 9, "▄▄", theme.muted);
          text(dripper_x - 1, 8, "(▓▓)╮", theme.muted);
          text(dripper_x + 3, 7, "│", theme.animation[0], true);
        } else {
          text(kettle_x + 1, 4, "▄▄", theme.muted);
          text(kettle_x, 3, "(▓▓)╮", theme.muted);
        }

        /* The barista, walking, working, or enjoying a cup: */
        const int x = barista_;
        const bool walking = barista_ != station();
        text(x, 5, "  O", plain, true);
        if (phase_ == Phase::done && !walking) {
          text(x + 4, 5, "c[]", plain);
          text(x + 5, 6, step_ % 8 < 4 ? "~" : "≀", plain, false, true);
          text(x, 4, " /|_/", plain);
          text(x, 3, "/ |", plain);
        } else if (pouring()) {
          text(x + 4, 5, "/", plain);
          text(x, 4, " /|/", plain);
          text(x, 3, "/ |", plain);
        } else {
          text(x, 4, " /|\\", plain);
          text(x, 3, "/ | \\", plain);
        }
        text(x, 2, "  |", plain);
        if (walking && step_ % 4 < 2) {
          text(x, 1, "  |", plain);
          text(x, 0, " / \\", plain);
        } else {
          text(x, 1, " / \\", plain);
          text(x, 0, "/   \\", plain);
        }

        repaint(stage, region, frame_, std::move(frame));
      }

      std::optional<Cell> origin_;

      std::string_view roaster_ = pick_roaster();

      Phase phase_ = Phase::weigh;
      /* The steps of wall time, and the steps of the animation: */
      int ticks_ = 0;
      int step_ = 0;
      int phase_step_ = 0;

      /* The column of the barista, starting off the bar to the left: */
      int barista_ = barista_start;

      /* The wall time of the first step: */
      std::optional<std::chrono::steady_clock::time_point> start_;

      /* The progress of the brew after the bloom, from 0 to 1: */
      double brew_ = 0.;

      /* The fill level of the pot from 0 to 1: */
      double level_ = 0.;

      /* The water poured so far in grams: */
      double water_ = 0.;

      std::vector<Glyph> frame_;
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
    case DashboardAnimation::eric:
      return std::make_unique<Coffee>();
    }

    __builtin_unreachable();
  }
} // namespace ryujin::animation
