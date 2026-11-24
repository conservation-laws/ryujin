//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <dashboard/animation.h>
#include <dashboard/theme.h>
#include <ryujin/base/random.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <format>
#include <map>
#include <numbers>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

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

    const auto brown = ftxui::Color::RGB(150, 95, 55);

    int free_area(const Stage &stage)
    {
      int area = 0;
      for (int y = 0; y < stage.screen.dimy(); ++y)
        for (int x = 0; x < stage.screen.dimx(); ++x)
          area += stage.free(x, y);
      return area;
    }

    int nearest(double x)
    {
      return static_cast<int>(std::lround(x));
    }

    /* Return @p text repeated @p n times: */
    std::string repeat(std::string_view text, int n)
    {
      std::string result;
      for (int i = 0; i < n; ++i)
        result += text;
      return result;
    }

    /** The content of a cell of a frame, empty if the cell is blank. */
    struct Glyph {
      std::string character;
      ftxui::Color color;
      bool bold = false;
      bool dim = false;
      ftxui::Color background = ftxui::Color::Default;

      bool operator==(const Glyph &) const = default;
    };

    /** The glyphs of the whole stage, row by row. */
    using Frame = std::vector<Glyph>;

    /*
     * Paint the cells of @p frame that differ from the @p last frame, and
     * make @p frame the last frame:
     */
    void repaint(Stage &stage, Frame &last, Frame frame)
    {
      const int width = stage.screen.dimx();
      if (last.size() != frame.size())
        last.assign(frame.size(), Glyph());

      for (std::size_t i = 0; i < frame.size(); ++i) {
        const auto &g = frame[i];
        if (g == last[i])
          continue;
        const Cell c{int(i) % width, int(i) / width};
        if (g.character.empty()) {
          stage.pixel(c.first, c.second);
        } else {
          paint(stage, c, g.character, g.color, g.bold, g.dim);
          stage.screen.PixelAt(c.first, c.second).background_color =
              g.background;
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
        const int area = free_area(stage);
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


    /* Return a random number in [a, b]: */
    int integer(int a, int b)
    {
      return std::uniform_int_distribution<int>(a, b)(random);
    }

    double uniform(double a, double b)
    {
      return std::uniform_real_distribution<double>(a, b)(random);
    }

    /* Move @p x towards @p target by at most @p speed: */
    double toward(double x, double target, double speed)
    {
      if (std::abs(target - x) <= speed)
        return target;
      return x + std::copysign(speed, target - x);
    }

    /* Split @p text into its UTF-8 characters: */
    std::vector<std::string_view> characters(std::string_view text)
    {
      std::vector<std::string_view> result;
      for (std::size_t i = 0; i < text.size();) {
        const auto lead = static_cast<unsigned char>(text[i]);
        const std::size_t n = lead < 0x80   ? 1
                              : lead < 0xe0 ? 2
                              : lead < 0xf0 ? 3
                                            : 4;
        result.push_back(text.substr(i, n));
        i += n;
      }
      return result;
    }

    using Rows = std::vector<std::string>;

    /* Mirror @p rows, padded to the same width, left to right: */
    Rows mirror(const Rows &rows)
    {
      static const std::map<std::string_view, std::string_view> swap = {
          {"(", ")"},
          {")", "("},
          {"/", "\\"},
          {"\\", "/"},
          {"<", ">"},
          {">", "<"},
          {"[", "]"},
          {"]", "["},
          {"{", "}"},
          {"}", "{"},
          {"╱", "╲"},
          {"╲", "╱"},
          {"`", "'"},
          {"'", "`"},
          {"╭", "╮"},
          {"╮", "╭"},
          {"╰", "╯"},
          {"╯", "╰"}};
      std::size_t width = 0;
      for (const auto &row : rows)
        width = std::max(width, characters(row).size());
      Rows result;
      for (const auto &row : rows) {
        auto glyphs = characters(row);
        glyphs.resize(width, " ");
        std::string mirrored;
        for (auto it = glyphs.rbegin(); it != glyphs.rend(); ++it) {
          const auto swapped = swap.find(*it);
          mirrored += swapped != swap.end() ? swapped->second : *it;
        }
        result.push_back(std::move(mirrored));
      }
      return result;
    }


    /*
     * A frame of the whole stage, composed back to front before it is
     * painted with repaint(). Only the free cells take a glyph.
     */
    struct Canvas {
      explicit Canvas(const Stage &stage)
          : stage(stage)
          , width(stage.screen.dimx())
          , cells(width * stage.screen.dimy())
      {
      }

      /*
       * Write @p text at the real-valued (@p x, @p y) into the free cells
       * within @p clip, skipping blanks:
       */
      void put(double x,
               double y,
               std::string_view text,
               ftxui::Color color,
               bool bold = false,
               bool dim = false,
               const ftxui::Box *clip = nullptr)
      {
        const int row = int(std::lround(y));
        int column = int(std::floor(x));
        for (const auto glyph : characters(text)) {
          if (glyph != " " && open(stage, column, row) &&
              (!clip || clip->Contain(column, row)))
            cells[row * width + column] = {
                std::string(glyph), color, bold, dim};
          ++column;
        }
      }

      /* Write the glyph @p g into the cell (@p x, @p y), if it is free: */
      void put(int x, int y, Glyph g)
      {
        if (open(stage, x, y))
          cells[y * width + x] = std::move(g);
      }

      /* Draw @p rows ending on row @p bottom: */
      void sprite(double x,
                  int bottom,
                  const Rows &rows,
                  ftxui::Color color,
                  bool dim = false,
                  const ftxui::Box *clip = nullptr)
      {
        for (std::size_t i = 0; i < rows.size(); ++i) {
          const int row = bottom - int(rows.size() - 1 - i);
          put(x, row, rows[i], color, false, dim, clip);
        }
      }

      /*
       * Draw an opaque sprite that also blanks the gaps inside each row,
       * so that it hides whatever is behind it:
       */
      void figure(double x,
                  int bottom,
                  const Rows &rows,
                  ftxui::Color color,
                  bool dim = false)
      {
        const auto blank = [](std::string_view g) { return g == " "; };
        for (std::size_t i = 0; i < rows.size(); ++i) {
          const int row = bottom - int(rows.size() - 1 - i);
          const auto glyphs = characters(rows[i]);
          const auto first = std::ranges::find_if_not(glyphs, blank);
          const auto last =
              std::ranges::find_if_not(glyphs.rbegin(), glyphs.rend(), blank)
                  .base();
          for (auto it = first; it < last; ++it) {
            const int column = int(std::floor(x)) + int(it - glyphs.begin());
            if (open(stage, column, row))
              cells[row * width + column] = Glyph();
          }
        }
        sprite(x, bottom, rows, color, dim);
      }

      const Stage &stage;
      const int width;
      Frame cells;
    };


    /**
     * The cafe around the bar of Coffee: the room in one-point perspective
     * with a street behind the window in the back wall (walkers, cyclists,
     * cars, a pigeon, and the weather), rows of tables down the depth of
     * the floor, and next to the bar a record player, a regular with a
     * laptop, a cat, and customers who queue, get a cup, and sit or leave.
     */
    class Cafe
    {
    public:
      enum class Weather : std::uint8_t { rain, snow, clear };

      /*
       * Lay out the room on the @p stage around the dashboard, with the
       * bottom left corner of the bar of @p bar_width x @p bar_height at
       * @p bar in the free @p band, and start with the room in full swing,
       * about four minutes in:
       */
      Cafe(const Stage &stage,
           const ftxui::Box &band,
           Cell bar,
           int bar_width,
           int bar_height,
           Weather weather)
          : weather_(weather)
      {
        const int width = stage.screen.dimx();
        const auto &box = stage.dashboard;
        const bool dashboard = box.x_min <= box.x_max;

        floor_ = bar.second;
        const int bar_top = floor_ - bar_height;

        /*
         * The back wall goes at the top of the screen, unless the
         * dashboard covers it; then right below the dashboard, if that is
         * above the bar:
         */
        top_ = dashboard && box.y_min < 7 && box.y_max < bar_top ? box.y_max + 1
                                                                 : 0;
        int bottom = bar_top;
        if (dashboard && top_ < box.y_min && box.y_max < bar_top)
          bottom = std::min(bottom, box.y_min);
        const int glass = std::clamp(bottom - top_ - 6, 2, 5);

        /*
         * One-point perspective: the back wall spans the rows from the
         * ceiling to the horizon, inset by inset_ on both sides, and the
         * side walls run from its corners towards the corners of the
         * screen at one column per row:
         */
        ceiling_ = top_ + 1;
        horizon_ = ceiling_ + 2 + glass + 2;
        inset_ = std::max(4, std::min(floor_ - horizon_, (width - 60) / 2));
        right_ = width - 1 - inset_;
        window_ = {inset_ + 3,
                   width - 1 - inset_ - 3,
                   ceiling_ + 1,
                   ceiling_ + 2 + glass};
        inside_ = {window_.x_min + 1,
                   window_.x_max - 1,
                   window_.y_min + 1,
                   window_.y_max - 1};
        mullions_ = {inside_.x_min + (inside_.x_max - inside_.x_min) / 3,
                     inside_.x_min + 2 * (inside_.x_max - inside_.x_min) / 3};

        door_ = band.x_max + 1;
        pickup_ = bar.first + bar_width + 1;
        /*
         * The queue holds three people, five columns apart, and the tables
         * (seven columns wide, with the seated customer) start right
         * after:
         */
        for (int x = pickup_ + 18; x < door_ - 7; x += 12)
          tables_.push_back(x);
        taken_.assign(tables_.size(), false);
        left_ = band.x_min;
        left_limit_ = bar.first - 2;
        /* The record player, the regular, and the cat only where they fit: */
        player_ = left_ + 9 < left_limit_;
        if (left_ + 24 < left_limit_)
          laptop_ = left_ + 12;

        /*
         * Lamps and fans hang in front of the side walls, the lamps lower,
         * so that they look closer, and only where the side walls leave
         * room for them:
         */
        const int lamp = std::max(1, int(std::lround(0.3 * inset_)) - 1);
        const int fan = std::max(1, int(std::lround(0.7 * inset_)) - 3);
        if (inset_ >= 5)
          lamps_ = {lamp, width - 1 - lamp - 2};
        if (fan + 7 < lamp || (lamp + 4 < fan && fan + 7 < inset_))
          fans_ = {fan, width - 1 - fan - 6};

        /*
         * Rows of tables down the depth of the floor, every eight rows
         * from just in front of the back wall to just behind the bar. Rows
         * in the back half are smaller and faded. People enter and leave
         * where the right side wall meets their row, or at the edge of the
         * screen:
         */
        const double middle = (horizon_ + floor_) / 2.;
        for (int y = horizon_ + 3;
             y + 3 < floor_ - bar_height && rows_.size() < 6;
             y += 8) {
          const int depth = y - horizon_;
          const int wall_left = std::max(0, inset_ - depth);
          const int wall_right =
              std::min(width - 1, width - 1 - inset_ + depth);
          Row row{.y = y, .near = y >= middle, .door = double(wall_right)};
          for (int x = wall_left + 4 + integer(0, 8);
               x + 9 < wall_right && row.tables.size() < 8;
               x += integer(16, 26))
            row.tables.push_back(x);
          row.seats.assign(row.tables.size(), {false, false});
          rows_.push_back(std::move(row));
        }

        cat_.x = std::max(left_ + 10, (left_ + left_limit_) / 2 - 4);

        for (int i = 0; i < 200; ++i)
          step_street();
        for (int i = 0; i < 1600; ++i) {
          step_room();
          step_rows();
        }
      }

      void step()
      {
        ++t_;
        step_street();
        step_room();
        step_rows();
      }

      void draw(Canvas &canvas) const;

    private:
      /* A walker, a cyclist, or a car on the street: */
      struct Traffic {
        enum class Kind : std::uint8_t { walker, bike, car } kind;
        int direction;
        double speed;
        double x;
        int width;
        ftxui::Color color;
        bool umbrella;
      };

      struct Pigeon {
        double x;
        double y;
        int direction;
        bool flying;
        int timer;
      };

      /* A note from the record player, or a "z" of the sleeping cat: */
      struct Float {
        double x;
        double y;
        int age;
        double phase;
        std::string_view glyph;
        ftxui::Color color;
      };

      struct Cat {
        enum class State : std::uint8_t { walk, sit, sleep };
        double x = 0.;
        double target = 0.;
        State state = State::sit;
        int timer = 60;
        int direction = 1;
      };

      /* A customer of the bar, with the index of a table, or -1: */
      struct Customer {
        enum class State : std::uint8_t { queue, served, seated, leaving };
        double x;
        State state;
        int timer;
        int order;
        int seat;
        ftxui::Color color;
        bool cup;
      };

      /* A customer at a table of a row, on side 0 or 1 of the table: */
      struct Guest {
        enum class State : std::uint8_t { walk, sit, leave };
        double x;
        int table;
        int side;
        State state;
        double target;
        int timer;
        bool laptop;
        ftxui::Color color;
      };

      struct Row {
        int y;
        bool near;
        double door;
        std::vector<int> tables = {};
        std::vector<std::array<bool, 2>> seats = {};
        std::vector<Guest> guests = {};
      };

      void spawn(Traffic::Kind kind, double speed, int width)
      {
        const int direction = pick(2) ? 1 : -1;
        traffic_.push_back(
            {kind,
             direction,
             speed,
             double(direction > 0 ? inside_.x_min - width : inside_.x_max + 1),
             width,
             random_color(),
             weather_ == Weather::rain && chance(0.7)});
      }

      void step_street()
      {
        const int x0 = inside_.x_min;
        const int x1 = inside_.x_max;
        const auto count = [this](Traffic::Kind kind) {
          return std::ranges::count_if(
              traffic_, [kind](const Traffic &a) { return a.kind == kind; });
        };
        if (count(Traffic::Kind::walker) < std::max(2, (x1 - x0) / 18) &&
            chance(0.08))
          spawn(Traffic::Kind::walker, uniform(0.15, 0.3), 3);
        if (count(Traffic::Kind::bike) == 0 && chance(0.008))
          spawn(Traffic::Kind::bike, uniform(0.6, 0.9), 3);
        if (count(Traffic::Kind::car) == 0 && chance(0.004))
          spawn(Traffic::Kind::car, uniform(1.0, 1.4), 5);
        for (auto &a : traffic_)
          a.x += a.direction * a.speed;
        std::erase_if(traffic_, [x0, x1](const Traffic &a) {
          return a.x < x0 - a.width - 1 || a.x > x1 + 1;
        });

        /* The weather on the glass: */
        switch (weather_) {
        case Weather::rain:
          for (auto &[x, y] : drops_) {
            x -= 1.;
            y += 1.;
          }
          std::erase_if(drops_, [&](const auto &p) {
            return p.first < x0 || p.second > inside_.y_max;
          });
          while (int(drops_.size()) < (x1 - x0) * 5 / 6)
            drops_.emplace_back(uniform(x0, x1 + 5),
                                inside_.y_min + uniform(-0.5, 0.5));
          break;
        case Weather::snow:
          for (auto &[x, y] : drops_) {
            x += 0.3 * (pick(3) - 1);
            y += 0.2;
          }
          std::erase_if(drops_, [&](const auto &p) {
            return p.first < x0 || p.first > x1 ||
                   p.second > inside_.y_max + 0.4;
          });
          while (int(drops_.size()) < (x1 - x0) / 5)
            drops_.emplace_back(uniform(x0, x1), double(inside_.y_min));
          break;
        case Weather::clear:
          /* Now and then a few birds fly across the clear sky: */
          for (auto &p : drops_)
            p.first += 0.5;
          std::erase_if(drops_, [x1](const auto &p) { return p.first > x1; });
          if (drops_.empty() && chance(0.01))
            for (int i = 0, n = integer(2, 4); i < n; ++i)
              drops_.emplace_back(x0 - 2. - 3 * i, inside_.y_min + i % 2);
          break;
        }

        /* The pigeon hops about, and flies off when someone comes near: */
        if (!pigeon_) {
          if (--pigeon_wait_ <= 0 && x1 - 4 > x0 + 3)
            pigeon_ = Pigeon{double(integer(x0 + 3, x1 - 5)),
                             double(inside_.y_max),
                             pick(2) ? 1 : -1,
                             false,
                             20};
          return;
        }
        auto &pigeon = *pigeon_;
        if (pigeon.flying) {
          pigeon.x += pigeon.direction;
          pigeon.y -= 0.4;
          if (pigeon.y < inside_.y_min || pigeon.x < x0 || pigeon.x > x1) {
            pigeon_.reset();
            pigeon_wait_ = integer(100, 300);
          }
          return;
        }
        if (std::ranges::any_of(traffic_, [&](const Traffic &a) {
              return std::abs(a.x + a.width / 2. - pigeon.x) < a.width / 2. + 3;
            })) {
          pigeon.flying = true;
          return;
        }
        if (--pigeon.timer <= 0) {
          pigeon.timer = integer(8, 40);
          if (chance(0.3))
            pigeon.direction *= -1;
          pigeon.x = std::clamp(pigeon.x + pigeon.direction, x0 + 1., x1 - 3.);
        }
      }

      void step_room()
      {
        static constexpr std::array<std::string_view, 2> notes = {"♪", "♫"};
        if (player_ && t_ % 7 == 0 && chance(0.8))
          notes_.push_back({left_ + 4.,
                            floor_ - 4.,
                            0,
                            uniform(0., 6.3),
                            notes[pick(2)],
                            random_color()});
        for (auto &note : notes_) {
          ++note.age;
          note.y -= 0.18;
          note.x += 0.18 + 0.25 * std::sin(0.3 * note.age + note.phase);
        }
        std::erase_if(notes_, [this](const Float &note) {
          return note.age >= 45 || note.y <= horizon_ - 3;
        });

        /* The cat walks to a spot, sits and blinks, or curls up for a nap: */
        const int lo = left_ + 10;
        const int hi = left_limit_ - 9; /* clear of the record player */
        if (hi > lo) {
          if (cat_.state == Cat::State::walk) {
            cat_.x = toward(cat_.x, cat_.target, 0.3);
            if (cat_.x == cat_.target) {
              cat_.state = chance(0.35) ? Cat::State::sleep : Cat::State::sit;
              cat_.timer = cat_.state == Cat::State::sleep ? integer(400, 900)
                                                           : integer(40, 160);
            }
          } else {
            --cat_.timer;
            if (cat_.state == Cat::State::sleep && t_ % 14 == 0)
              zs_.push_back({cat_.x + 7, floor_ - 1., 0, 0., "z", {}});
            if (cat_.timer <= 0) {
              cat_.state = Cat::State::walk;
              cat_.target = integer(lo, hi);
              cat_.direction = cat_.target > cat_.x ? 1 : -1;
            }
          }
        }
        for (auto &z : zs_) {
          ++z.age;
          z.y -= 0.15;
          z.x += 0.1;
        }
        std::erase_if(zs_, [](const Float &z) { return z.age >= 26; });

        /* Customers come in from the right, queue, get a cup, sit or leave: */
        using State = Customer::State;
        const auto queued = [](const Customer &c) {
          return c.state == State::queue;
        };
        const bool door_clear =
            std::ranges::all_of(customers_, [&](const Customer &c) {
              return !queued(c) || c.x < door_ - 6;
            });
        if (std::ranges::count_if(customers_, queued) < 3 &&
            customers_.size() < 6 && pickup_ + 14 < door_ && door_clear &&
            chance(0.02))
          customers_.push_back({double(door_),
                                State::queue,
                                0,
                                integer(25, 60),
                                -1,
                                random_color(),
                                false});

        int k = 0;
        for (auto &c : customers_) {
          if (!queued(c))
            continue;
          const double target = pickup_ + 5 * k;
          c.x = toward(c.x, target, 0.4);
          if (k++ == 0 && c.x == target && ++c.timer >= c.order) {
            c.state = State::served;
            c.cup = true;
            std::vector<int> free;
            for (std::size_t i = 0; i < taken_.size(); ++i)
              if (!taken_[i])
                free.push_back(int(i));
            if (!free.empty() && chance(0.6)) {
              c.seat = free[pick(free.size())];
              taken_[c.seat] = true;
            }
          }
        }

        for (auto &c : customers_) {
          switch (c.state) {
          case State::served: {
            const double target =
                c.seat >= 0 ? double(tables_[c.seat]) : door_ + 1.;
            c.x = toward(c.x, target, 0.4);
            if (c.seat >= 0 && c.x == target) {
              c.state = State::seated;
              c.timer = integer(250, 700);
            }
            break;
          }
          case State::seated:
            if (--c.timer <= 0) {
              taken_[c.seat] = false;
              c.state = State::leaving;
              c.cup = false;
              c.seat = -1;
            }
            break;
          case State::leaving:
            c.x = toward(c.x, door_ + 1., 0.4);
            break;
          case State::queue:
            break;
          }
        }
        std::erase_if(customers_,
                      [this](const Customer &c) { return c.x >= door_ + 1; });
      }

      /*
       * Customers in the rows of tables come in from the right, sit at a
       * table, on either side in the back rows, and leave the same way:
       */
      void step_rows()
      {
        using State = Guest::State;
        for (auto &row : rows_) {
          std::vector<std::pair<int, int>> free;
          for (std::size_t i = 0; i < row.seats.size(); ++i)
            for (int side = 0; side < (row.near ? 1 : 2); ++side)
              if (!row.seats[i][side])
                free.emplace_back(int(i), side);
          if (!free.empty() && chance(0.006)) {
            const auto [i, side] = free[pick(free.size())];
            const int x = row.tables[i];
            const int target = row.near ? x - 4 : side == 0 ? x - 2 : x + 2;
            /* Shorter stays at the front, so that its rows never fill up: */
            row.guests.push_back(
                {row.door,
                 i,
                 side,
                 State::walk,
                 double(target),
                 row.near ? integer(150, 450) : integer(300, 900),
                 row.near && chance(0.35),
                 random_color()});
            row.seats[i][side] = true;
          }
          const double speed = row.near ? 0.4 : 0.35;
          for (auto &guest : row.guests) {
            switch (guest.state) {
            case State::walk:
              guest.x = toward(guest.x, guest.target, speed);
              if (guest.x == guest.target)
                guest.state = State::sit;
              break;
            case State::sit:
              if (--guest.timer <= 0) {
                row.seats[guest.table][guest.side] = false;
                guest.state = State::leave;
              }
              break;
            case State::leave:
              guest.x = toward(guest.x, row.door + 2., speed);
              break;
            }
          }
          std::erase_if(row.guests, [&row](const Guest &guest) {
            return guest.x >= row.door + 2.;
          });
        }
      }

      Weather weather_;
      int t_ = 0;

      /* The layout of the room: */
      int floor_ = 0;
      int top_ = 0;
      int ceiling_ = 0;
      int horizon_ = 0;
      int inset_ = 0;
      int right_ = 0;
      ftxui::Box window_ = {};
      ftxui::Box inside_ = {};
      std::array<int, 2> mullions_ = {};
      std::vector<int> lamps_;
      std::vector<int> fans_;

      /* The layout of the floor next to the bar: */
      int door_ = 0;
      int pickup_ = 0;
      int left_ = 0;
      int left_limit_ = 0;
      bool player_ = false;
      std::optional<int> laptop_;
      std::vector<int> tables_;
      std::vector<bool> taken_;
      std::vector<Row> rows_;

      std::vector<Traffic> traffic_;
      std::vector<std::pair<double, double>> drops_;
      std::optional<Pigeon> pigeon_;
      int pigeon_wait_ = 0;
      std::vector<Float> notes_;
      std::vector<Float> zs_;
      Cat cat_;
      std::vector<Customer> customers_;
    };


    void Cafe::draw(Canvas &canvas) const
    {
      const auto &theme = dashboard::theme;
      const auto muted = theme.muted;
      const auto plain = ftxui::Color::Default;
      const auto ginger = ftxui::Color::RGB(255, 175, 95);
      const int t = t_;

      /* The room, faded because it is far, and the street behind the glass: */
      const auto far = [&](double x, double y, std::string_view text) {
        canvas.put(x, y, text, muted, false, true);
      };
      const auto street =
          [&](double x, double y, std::string_view text, ftxui::Color color) {
            canvas.put(x, y, text, color, false, true, &inside_);
          };
      /* Outline @p b with the corners and edges @p glyphs, like "┌─┐│└┘": */
      const auto outline = [&](const ftxui::Box &b, std::string_view glyphs) {
        const auto g = characters(glyphs);
        const auto edge = repeat(g[1], b.x_max - b.x_min - 1);
        far(b.x_min, b.y_min, std::string(g[0]) + edge + std::string(g[2]));
        for (int y = b.y_min + 1; y < b.y_max; ++y) {
          far(b.x_min, y, g[3]);
          far(b.x_max, y, g[3]);
        }
        far(b.x_min, b.y_max, std::string(g[4]) + edge + std::string(g[5]));
      };

      /* The room in one-point perspective: */
      outline({inset_, right_, ceiling_, horizon_}, "┌─┐│└┘");
      for (int k = 1; k <= ceiling_ - top_; ++k) {
        far(inset_ - k, ceiling_ - k, "╲");
        far(right_ + k, ceiling_ - k, "╱");
      }
      for (int k = 1; k <= floor_ - horizon_; ++k) {
        far(inset_ - k, horizon_ + k, "╱");
        far(right_ + k, horizon_ + k, "╲");
      }

      /* The window in the back wall, with its sill: */
      outline(window_, "╔═╗║╚╝");
      canvas.put(window_.x_min - 1,
                 window_.y_max + 1,
                 repeat("▀", window_.x_max - window_.x_min + 3),
                 brown,
                 false,
                 true);

      for (const auto &[x, y] : drops_)
        switch (weather_) {
        case Weather::rain:
          street(x, y, "╱", theme.animation[0]);
          break;
        case Weather::snow:
          street(x, y, int(x * 7) % 3 ? "*" : "·", plain);
          break;
        case Weather::clear:
          street(x, y, (t / 3) % 2 ? "v" : "^", muted);
          break;
        }

      for (const auto &a : traffic_) {
        Rows rows;
        switch (a.kind) {
        case Traffic::Kind::walker:
          rows = {" o ", int(a.x * 2) % 2 ? " Λ " : " λ "};
          if (a.umbrella)
            rows.insert(rows.begin(), "╭┴╮");
          break;
        case Traffic::Kind::bike:
          rows = {"  o", "o'o"};
          break;
        case Traffic::Kind::car:
          rows = {" ▄▄▄ ", "▀o▀o▀"};
          break;
        }
        if (a.direction < 0)
          rows = mirror(rows);
        canvas.sprite(a.x, inside_.y_max, rows, a.color, true, &inside_);
      }

      if (pigeon_) {
        const auto &pigeon = *pigeon_;
        const auto glyph = pigeon.flying          ? (t % 2 ? "\\v/" : "‾v‾")
                           : pigeon.direction > 0 ? "(o>"
                                                  : "<o)";
        street(pigeon.x, pigeon.y, glyph, muted);
      }

      for (const int x : mullions_)
        for (int y = inside_.y_min; y <= inside_.y_max; ++y)
          far(x, y, "│");

      const Rows table = {"═╤═", " │"};
      /* Seated, facing right, a hand on the table, the knees under it: */
      const Rows seated = {" o", "/|\\_", " └┐"};
      /* Seated at a table with a steaming cup of coffee: */
      const auto drinker = [&](double x, int y, ftxui::Color color, int phase) {
        canvas.figure(x, y, seated, color);
        canvas.put(x + 4, y - 2, "c[]", color);
        canvas.put(
            x + 5, y - 3, (t / 4 + phase) % 2 ? "~" : "≀", color, false, true);
      };

      /*
       * The rows of tables, back to front. People walking pass behind the
       * tables and the people sitting at them:
       */
      for (const auto &row : rows_) {
        const int y = row.y;
        for (const auto &guest : row.guests) {
          if (guest.state == Guest::State::sit)
            continue;
          const bool step = int(guest.x * (row.near ? 1.3 : 2.)) % 2;
          if (row.near)
            canvas.figure(
                guest.x, y, {" o", "/|\\", step ? "/ \\" : " | "}, guest.color);
          else
            canvas.figure(
                guest.x, y, {" o ", step ? " Λ " : " λ "}, guest.color, true);
        }
        for (const int x : row.tables)
          if (row.near)
            canvas.figure(x, y, table, muted);
          else
            far(x, y, "─┬─");
        for (const auto &guest : row.guests) {
          if (guest.state != Guest::State::sit)
            continue;
          const int x = row.tables[guest.table];
          if (!row.near) {
            canvas.sprite(guest.x, y, {" o ", " Λ "}, guest.color, true);
            const int cup = guest.side == 0 ? x : x + 2;
            canvas.put(cup, y - 1, "u", guest.color, false, true);
            if ((t / 5 + guest.table + guest.side) % 3 == 0)
              canvas.put(cup, y - 2, "~", guest.color, false, true);
          } else if (guest.laptop) {
            canvas.figure(guest.x, y, seated, guest.color);
            canvas.put(x, y - 2, (t + guest.table) % 4 < 2 ? "▁" : "▂", muted);
            canvas.put(x + 1, y - 2, "╱", theme.caption, true);
          } else {
            drinker(guest.x, y, guest.color, guest.table);
          }
        }
      }

      /* Pendant lamps, hanging from the ceiling, that flicker now and then: */
      for (std::size_t i = 0; i < lamps_.size(); ++i) {
        const int x = lamps_[i];
        const int shade = horizon_ - 4;
        for (int y = top_; y < shade; ++y)
          canvas.put(x + 1, y, "│", muted);
        /* A bell-shaped shade, with the bulb peeking out right below: */
        canvas.put(x, shade, "▄█▄", brown);
        const bool flicker = (t + 37 * int(i)) % 97 < 2;
        canvas.put(x + 1,
                   shade + 1,
                   "▀",
                   ftxui::Color::RGB(255, 175, 0),
                   !flicker,
                   flicker);
      }

      /*
       * Ceiling fans seen from the side: the blades shorten and lengthen
       * as they turn, one step per frame:
       */
      static constexpr std::array<std::string_view, 4> blades = {
          "━━━┿━━━", " ━━┿━━ ", "  ━┿━  ", " ━━┿━━ "};
      for (std::size_t i = 0; i < fans_.size(); ++i) {
        const int x = fans_[i];
        for (int y = top_; y < top_ + 3; ++y)
          canvas.put(x + 3, y, "│", muted);
        canvas.put(x, top_ + 3, blades[(t + 2 * i) % 4], muted);
      }

      /* The record player, spinning: */
      static constexpr std::array<std::string_view, 4> platters = {
          "◐", "◓", "◑", "◒"};
      if (player_)
        canvas.sprite(left_ + 1,
                      floor_,
                      {"╭──────╮",
                       "│(" + std::string(platters[(t / 2) % 4]) + ") ╲ │",
                       "├──────┤",
                       "│▌▌▌▌▌▌│"},
                      muted);
      for (const auto &note : notes_)
        canvas.put(
            note.x, note.y, note.glyph, note.color, false, note.age > 28);

      /* The regular, seated at a table with a laptop, typing: */
      if (laptop_) {
        const int x = *laptop_;
        canvas.sprite(x + 4, floor_, {"═══╤═══", "   │"}, muted);
        canvas.sprite(x, floor_, seated, plain);
        canvas.put(x + 5, floor_ - 2, t % 4 < 2 ? "▁▂" : "▂▁", muted);
        canvas.put(x + 7, floor_ - 2, "╱", theme.caption, true);
      }

      /* The cat: */
      if (left_limit_ - 9 > left_ + 10) {
        const bool closed = cat_.state == Cat::State::sleep ||
                            (cat_.state == Cat::State::sit && t % 40 < 2);
        canvas.sprite(cat_.x,
                      floor_,
                      {" /\\_/\\", closed ? "( -.- )" : "( o.o )"},
                      ginger);
        canvas.put(cat_.x + (cat_.direction > 0 ? 7 : -1),
                   floor_,
                   (t / 3) % 2 ? "~" : "-",
                   ginger);
        for (const auto &z : zs_)
          canvas.put(z.x, z.y, z.age < 13 ? "z" : "Z", ginger, false, true);
      }

      /*
       * Back to front: people walking pass behind the tables, and the
       * queue stands in front of everyone walking by:
       */
      using State = Customer::State;
      const auto person = [&](const Customer &c) {
        const auto legs =
            int(c.x * 1.3) % 2 || (c.state == State::queue && c.timer) ? "/ \\"
                                                                       : " | ";
        canvas.figure(
            c.x, floor_, {" o", c.cup ? "/|\\c[]" : "/|\\", legs}, c.color);
        if (c.cup)
          canvas.put(c.x + 4,
                     floor_ - 2,
                     (t / 4) % 2 ? "~" : "≀",
                     c.color,
                     false,
                     true);
      };
      const auto queued = [this](const Customer &c) {
        return c.state == State::queue && c.x <= pickup_ + 14;
      };
      for (const auto &c : customers_)
        if (c.state != State::seated && !queued(c))
          person(c);

      for (const int x : tables_)
        canvas.figure(x + 4, floor_, table, muted);
      for (const auto &c : customers_)
        if (c.state == State::seated)
          drinker(tables_[c.seat], floor_, c.color, 0);

      for (auto it = customers_.rbegin(); it != customers_.rend(); ++it)
        if (queued(*it))
          person(*it);
    }


    /**
     * A pour-over coffee bar in a Cafe around the dashboard: a barista
     * walks up, grinds the beans, weighs the grounds on the scale in the
     * counter under the dripper, and brews a pot of coffee that is
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
        frame_.clear();
        show(stage);
      }

    private:
      enum class Phase : std::uint8_t { grind, weigh, bloom, pour, done };

      /* The duration of the phases in steps of the animation: */
      static constexpr int grind_steps = 30;
      static constexpr int weigh_steps = 25;
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
      static constexpr int bar_width = 47;
      static constexpr int bar_height = 9;

      /* The columns of the stations, relative to the left end: */
      static constexpr int bag_x = 0;
      static constexpr int grinder_x = 15;
      static constexpr int dripper_x = 26;
      static constexpr int kettle_x = 34;

      /* The column of the barista before walking up to the bar: */
      static constexpr int barista_start = -6;

      /* The duration of a step of the animation: */
      static constexpr std::chrono::milliseconds step_duration{150};

      /*
       * The steps of the preparations before the pour: walking to the
       * stations at a column every two steps, grinding, weighing, and the
       * bloom:
       */
      static constexpr double preparation_steps =
          2 * (dripper_x - 5 - barista_start) + grind_steps + weigh_steps +
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

      /*
       * Place the bottom left corner of the bar in a free band below,
       * above, left, or right of the dashboard:
       */
      void place(Stage &stage)
      {
        const auto &box = stage.dashboard;
        const int width = stage.screen.dimx();
        const int height = stage.screen.dimy();

        const std::array<ftxui::Box, 4> bands = {{
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

            /* Lay out the cafe again only if the stage has changed: */
            const Layout layout{width, height, box, band};
            if (!cafe_ || layout != layout_) {
              layout_ = layout;
              cafe_.emplace(
                  stage, band, *origin_, bar_width, bar_height, weather_);
            }
            return;
          }
        }
        cafe_.reset();

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
        if (cafe_)
          cafe_->step();
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
        case Phase::grind:
          if (phase_step_ == grind_steps)
            enter(Phase::weigh);
          break;
        case Phase::weigh:
          if (phase_step_ == weigh_steps)
            enter(Phase::bloom);
          break;
        case Phase::bloom:
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
          break;
        }
        case Phase::done:
          break;
        }
      }

      /* The fill level of the pot from 0 to 1: */
      double level() const
      {
        if (phase_ < Phase::bloom)
          return 0.;
        if (phase_ == Phase::bloom)
          return 0.05 * phase_step_ / bloom_steps;
        return 0.05 + 0.95 * brew_;
      }

      /* The water poured so far in grams, from the bloom on: */
      double water() const
      {
        if (phase_ == Phase::bloom)
          return bloom_water * std::min(phase_step_, bloom_pour_steps) /
                 bloom_pour_steps;
        double water = bloom_water;
        for (const auto &pour : pours)
          water += pour.water *
                   std::clamp(
                       (brew_ - pour.begin) / (pour.end - pour.begin), 0., 1.);
        return water;
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

        const ftxui::Box bar = {origin_->first,
                                origin_->first + bar_width - 1,
                                origin_->second - bar_height + 1,
                                origin_->second};

        /* The frame covers the whole screen, the cafe behind the bar: */
        Canvas canvas(stage);
        if (cafe_)
          cafe_->draw(canvas);

        /* Write the text @p row rows above the bottom of the bar: */
        const auto text = [&](int column,
                              int row,
                              std::string_view text,
                              ftxui::Color color,
                              bool bold = false,
                              bool dim = false) {
          canvas.put(origin_->first + column,
                     origin_->second - row,
                     text,
                     color,
                     bold,
                     dim,
                     &bar);
        };

        const auto &theme = dashboard::theme;
        const auto plain = ftxui::Color::Default;
        const bool grinding = phase_ == Phase::grind && barista_ == station();

        /*
         * The display of the scale in the counter under the dripper: the
         * grounds while weighing, then the timer of the brew from the
         * start of the bloom, and the water poured so far. The timer runs
         * as if the brew took two and a half minutes, however long the
         * simulation takes:
         */
        const bool weighing = phase_ == Phase::weigh && barista_ == station();
        std::string display;
        if (phase_ >= Phase::bloom) {
          const int seconds =
              phase_ == Phase::bloom
                  ? bloom_seconds * phase_step_ / bloom_steps
                  : bloom_seconds +
                        int(std::lround((brew_seconds - bloom_seconds) *
                                        brew_));
          display = std::format(
              "{}:{:02} {:3.0f}g", seconds / 60, seconds % 60, water());
        } else {
          display = std::format(
              "{:4.1f}g", weighing ? 18. * phase_step_ / weigh_steps : 0.);
        }
        const int length = int(display.size());
        const int display_x = dripper_x + 3 - length / 2;

        /* The counter, with a gap around the display: */
        for (int x = 0; x < bar_width; ++x)
          if (x < display_x - 1 || x > display_x + length)
            text(x, 2, "━", theme.muted);
        text(display_x, 2, display, theme.good, true);
        for (const int x : {0, bar_width - 1})
          for (const int row : {0, 1})
            text(x, row, "┃", theme.muted);

        /* The bag of beans: */
        text(bag_x, 4, "┌~~~~~~~~┐", theme.muted);
        text(bag_x, 3, "│", theme.muted);
        text(bag_x + 1 + (8 - int(roaster_.size()) + 1) / 2,
             3,
             roaster_,
             brown,
             true);
        text(bag_x + 9, 3, "│", theme.muted);

        /* The grinder, shaking while it grinds: */
        const int shake = grinding ? step_ % 2 : 0;
        const bool hopper = phase_ == Phase::grind &&
                            (!grinding || phase_step_ < 2 * grind_steps / 3);
        text(grinder_x + 1 + shake, 6, "▄▄▄", theme.muted);
        text(grinder_x + 1 + shake, 5, "\\ /", theme.muted);
        if (hopper)
          text(grinder_x + 2 + shake,
               5,
               grinding && step_ % 2 ? "░" : "▒",
               brown);
        text(grinder_x + 1 + shake, 4, "|▓|", theme.muted);
        text(grinder_x, 3, "[___]", theme.muted);
        if (grinding)
          text(grinder_x - 1 + 2 * shake, 7, "brrr", theme.muted, false, true);

        /* The dripper on top of the pot, filled with the grounds: */
        text(dripper_x, 6, "\\     /", theme.muted);
        text(dripper_x + 1, 5, "\\___/", theme.muted);
        text(dripper_x, 4, "(     )", theme.muted);
        text(dripper_x, 3, "(     )", theme.muted);
        if (phase_ >= Phase::bloom) {
          const bool wet = pouring() || phase_ == Phase::pour;
          text(dripper_x + 1,
               6,
               wet && step_ % 4 < 2 ? "~~~~~" : "▒▒▒▒▒",
               wet ? theme.animation[0] : brown);
        } else if (weighing) {
          const int grounds = int(std::ceil(5. * phase_step_ / weigh_steps));
          text(dripper_x + 1, 6, repeat("▒", grounds), brown);
          if (phase_step_ < weigh_steps - 4)
            text(dripper_x + 3, 7, "∘", brown, true);
        }

        /* The coffee in the pot, in eighths of a cell: */
        static constexpr std::array<std::string_view, 9> blocks = {
            " ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
        const int eighths = int(std::lround(16. * level()));
        if (eighths > 0)
          text(
              dripper_x + 1, 3, repeat(blocks[std::min(eighths, 8)], 5), brown);
        if (eighths > 8)
          text(dripper_x + 1, 4, repeat(blocks[eighths - 8], 5), brown);
        if (phase_ >= Phase::bloom && phase_ != Phase::done && eighths < 16 &&
            step_ % 3 == 0)
          text(dripper_x + 3, 4, "·", brown, true);

        /* The kettle, pouring right into the dripper: */
        if (pouring()) {
          text(dripper_x, 8, "▄▄", theme.muted);
          text(dripper_x - 1, 7, "(▓▓)╮", theme.muted);
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

        repaint(stage, frame_, std::move(canvas.cells));
      }

      std::optional<Cell> origin_;

      /* The stage the cafe is laid out for: */
      struct Layout {
        int width;
        int height;
        ftxui::Box dashboard;
        ftxui::Box band;

        bool operator==(const Layout &) const = default;
      };
      Layout layout_ = {};
      std::optional<Cafe> cafe_;
      Cafe::Weather weather_ = Cafe::Weather(pick(3));

      std::string_view roaster_ = pick_roaster();

      Phase phase_ = Phase::grind;
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

      Frame frame_;
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

        std::erase_if(fish_, [&](const Fish &f) { return !fits(stage, f); });
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
        frame_.clear();
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
          return !open(stage, nearest(b.x), nearest(b.y));
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

      const std::vector<Rows> right_ = {
          Rows{"><>"},
          Rows{"><(((°>"},
          Rows{">(°)"},
          Rows{"  \\ | /  ", ">-(  ° )-", "  / | \\  "},
          Rows{"V(°.°)V"},
          Rows{"Y(°.°)Y"},
          Rows{" .--. ", "(    )", " )()( ", " (  ) ", " )  ( "},
          Rows{" .--. ", "(    )", " ()() ", " )  ( ", " (  ) "},
          Rows{" .-. ", "(   )", " ||| ", " | | ", " | | "}};
      const std::vector<Rows> left_ = [this] {
        std::vector<Rows> sprites;
        for (const auto &sprite : right_)
          sprites.push_back(mirror(sprite));
        return sprites;
      }();

      const Rows &look(const Fish &f) const
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

      bool fits(const Stage &stage, const Fish &f, double x) const
      {
        const auto &sprite = look(f);
        if (f.y + int(sprite.size()) > height_ - 1)
          return false;
        for (int r = 0; r < int(sprite.size()); ++r) {
          const auto glyphs = characters(sprite[r]);
          for (int k = 0; k < int(glyphs.size()); ++k)
            if (glyphs[k] != " " && !open(stage, nearest(x) + k, f.y + r))
              return false;
        }
        return true;
      }

      bool fits(const Stage &stage, const Fish &f) const
      {
        return fits(stage, f, f.x);
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
            fits(stage, f, x) && !chance(0.003))
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
            bubble(f.x + (f.dir > 0 ? characters(look(f)[0]).size() : -1.),
                   f.y);
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
        Canvas canvas(stage);

        const auto floor = dashboard::theme.animation.front();
        for (int x = 0; x < width_; ++x)
          canvas.put(
              x, height_ - 1, sand[x * 5 % sand.size()], floor, false, true);

        for (const auto &w : weeds_)
          for (int r = 0; r < w.height && open(stage, w.x, height_ - 2 - r);
               ++r)
            canvas.put(w.x,
                       height_ - 2 - r,
                       (time_ / 8 + r + w.phase) % 2 ? "(" : ")",
                       w.color);

        const auto bright = dashboard::theme.animation.back();
        for (const auto &b : bubbles_)
          canvas.put(nearest(b.x),
                     nearest(b.y),
                     b.age < 6    ? "."
                     : b.age < 16 ? "o"
                                  : "O",
                     bright,
                     false,
                     b.age < 6);

        for (const bool front : {false, true})
          for (const auto &f : fish_) {
            if ((f.kind == Kind::jelly || f.kind == Kind::crab) == front)
              continue;
            const auto &sprite = look(f);
            for (int r = 0; r < int(sprite.size()); ++r)
              canvas.put(nearest(f.x),
                         f.y + r,
                         sprite[r],
                         f.color,
                         true,
                         f.kind == Kind::jelly);
          }

        repaint(stage, frame_, std::move(canvas.cells));
      }

      int width_ = 0;
      int height_ = 0;
      int time_ = 0;
      std::vector<Fish> fish_;
      std::vector<Weed> weeds_;
      std::vector<Bubble> bubbles_;
      Frame frame_;
    };


    /**
     * A night in the forest: the lower two thirds of the stage are a
     * forest in pixel art, with three rows of pines, oaks and birches, a
     * cabin with a lit window, a pond with a frog, undergrowth, an owl,
     * and animals that roam the forest floor; the upper third is a night
     * sky with twinkling stars, drifting clouds, now and then a flock of
     * birds or a shooting star, and the moon, which crosses the sky and
     * waxes from a crescent to the full moon over the run. The scenery is
     * fixed: every plant and star sits at a fixed column, counted from the
     * left, so that a larger stage shows more of the same forest.
     *
     * The forest is drawn in pixels, two per cell, one above the other,
     * with half blocks.
     */
    class Forest final : public Animation
    {
    public:
      void draw(Stage &stage) override
      {
        progress_ = std::max(0., stage.progress);
        width_ = stage.screen.dimx();
        height_ = stage.screen.dimy();
        forest_height_ = height_ * 2 / 3;

        /* The scenery depends only on the size of the stage and the colors: */
        const auto planting = std::tuple{width_, height_, dashboard::theme};
        if (planted_ != planting) {
          plant();
          planted_ = planting;
        }

        if (clouds_.empty())
          for (int k = 0; k < std::max(1, width_ / 50); ++k)
            clouds_.push_back({.x = double(pick(width_)),
                               .y = 1 + pick(std::max(1, sky_rows() - 3)),
                               .seed = k,
                               .speed = 0.02 + 0.03 * pick(100) / 100.});
        if (animals_.empty())
          for (int kind = 0; kind < static_cast<int>(animal_bitmaps_.size());
               ++kind)
            animals_.push_back({.x = 0., .kind = kind, .right = chance(0.5)});

        /*
         * Place every animal that does not fit on the floor, at the start
         * or after a resize, at a free place, or hide it if there is none:
         */
        for (auto &a : animals_) {
          if (a.visible && free_floor(a.kind, a.x, &a))
            continue;
          a.visible = false;
          for (int attempt = 0; attempt < 64 && !a.visible; ++attempt) {
            const double x = pick(width_);
            if (free_floor(a.kind, x, &a)) {
              a.x = x;
              a.visible = true;
            }
          }
        }

        frame_.clear();
        compose(stage);
      }

      void step(Stage &stage) override
      {
        ++time_;
        progress_ = std::max(0., stage.progress);

        for (auto &c : clouds_) {
          c.x += c.speed;
          if (c.x > width_)
            c.x = -cloud_width;
        }

        /* Now and then a flock of birds crosses the sky: */
        if (!flock_ && chance(0.003))
          flock_ = Cell{-12 * scale(),
                        2 + pick(std::max(1, 2 * sky_rows() - 8 * scale()))};
        if (flock_ && time_ % 2 == 0 && ++flock_->first > width_)
          flock_.reset();

        if (meteors_.empty() && chance(0.004))
          meteors_.push_back({.x = double(pick(width_)),
                              .y = double(pick(std::max(1, sky_rows() / 2)))});
        std::erase_if(meteors_, [&](Meteor &m) {
          m.x += 1.5;
          m.y += 0.5;
          return ++m.age > 14 || m.x >= width_ || m.y >= sky_rows();
        });

        /* Smoke rises from the chimney and drifts off with the wind: */
        if (cabin_ && time_ % 6 == 0)
          smoke_.push_back(
              {.x = double(chimney_.first), .y = double(chimney_.second)});
        std::erase_if(smoke_, [&](Puff &p) {
          p.x += 0.12 + 0.05 * uniform(-1., 1.);
          p.y -= 0.15;
          return ++p.age > 40 || p.y < 0;
        });

        for (auto &a : animals_)
          roam(a);

        if (croak_ > 0)
          --croak_;
        else if (chance(0.01))
          croak_ = 8;

        if (owl_look_ != 0 ? chance(0.05) : chance(0.01))
          owl_look_ = owl_look_ != 0 ? 0 : chance(0.5) ? 1 : -1;

        compose(stage);
      }

    private:
      using Color = ftxui::Color;

      /*
       * A picture in pixels: every character is a pixel whose role picks
       * its color, see role_color(), and '.' is transparent.
       */
      using Bitmap = std::vector<std::string_view>;

      /*
       * A picture in pixels by (column, pixel row), two pixel rows per row
       * of cells, stored densely over a box. A pixel without a color is
       * transparent, and pixels outside the box are dropped:
       */
      class Pixels
      {
      public:
        Pixels() = default;

        explicit Pixels(ftxui::Box box)
            : box_(box)
            , width_(std::max(0, box.x_max - box.x_min + 1))
            , colors_(static_cast<std::size_t>(width_) *
                      std::max(0, box.y_max - box.y_min + 1))
        {
        }

        void set(Cell c, Color color)
        {
          if (box_.Contain(c.first, c.second))
            colors_[index(c)] = color;
        }

        /* The color of the pixel @p c, null if it is transparent: */
        const Color *find(Cell c) const
        {
          if (!box_.Contain(c.first, c.second))
            return nullptr;
          const auto &color = colors_[index(c)];
          return color ? &*color : nullptr;
        }

        bool contains(Cell c) const
        {
          return find(c) != nullptr;
        }

        /* Call @p f(c, color) for every pixel c with a color: */
        template <typename F>
        void for_each(F f) const
        {
          for (int y = box_.y_min; y <= box_.y_max; ++y)
            for (int x = box_.x_min; x <= box_.x_max; ++x)
              if (const auto *color = find({x, y}))
                f(Cell{x, y}, *color);
        }

        /*
         * Call @p f(x, y, top, low) for every cell (x, y) with a pixel of
         * a color, with the colors of its two pixels, null if transparent:
         */
        template <typename F>
        void for_each_cell(F f) const
        {
          for (int y = box_.y_min / 2; 2 * y <= box_.y_max; ++y)
            for (int x = box_.x_min; x <= box_.x_max; ++x) {
              const auto *top = find({x, 2 * y});
              const auto *low = find({x, 2 * y + 1});
              if (top || low)
                f(x, y, top, low);
            }
        }

      private:
        std::size_t index(Cell c) const
        {
          return static_cast<std::size_t>(c.second - box_.y_min) * width_ +
                 (c.first - box_.x_min);
        }

        ftxui::Box box_ = {0, -1, 0, -1};
        int width_ = 0;
        std::vector<std::optional<Color>> colors_;
      };

      struct Cloud {
        double x;
        int y;
        int seed;
        double speed;
      };

      struct Meteor {
        double x;
        double y;
        int age = 0;
      };

      struct Puff {
        double x;
        double y;
        int age = 0;
      };

      struct Animal {
        double x;
        int kind;
        bool right;
        int pause = 0;
        bool visible = false;
        bool hiding = false;
      };

      /* The kinds of animals, the indices into animal_bitmaps_: */
      static constexpr int deer = 0;
      static constexpr int bunny = 4;

      /* A fixed pseudo-random number for (x, layer): */
      static std::uint64_t hash(int x, int layer)
      {
        std::uint64_t z = 0x9e3779b97f4a7c15ull *
                          (static_cast<std::uint64_t>(x) * 64 + layer + 1);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
      }

      /* A fixed pseudo-random number in [0, 1) for (x, layer): */
      static double fraction(int x, int layer)
      {
        return static_cast<double>(hash(x, layer) >> 11) * 0x1.0p-53;
      }

      int sky_rows() const
      {
        return height_ - forest_height_;
      }

      /* The pixel row of the forest floor: */
      int ground() const
      {
        return 2 * height_ - 1;
      }

      ftxui::Box stage_pixels() const
      {
        return {0, width_ - 1, 0, ground()};
      }

      /*
       * The scale of every figure in pixels per pixel of its bitmap, so
       * that the deer, 16 pixels tall, is about a fifth of the forest:
       */
      int scale() const
      {
        return std::max(1, nearest(0.2 * 2 * forest_height_ / 16.));
      }

      /*
       * The color of a role: 'f' pine needles, 'l' leaves, 'a' autumn
       * leaves, 't' bark, 'b' light fur and birch bark, 'c' red caps,
       * berries and roof, 'o' fox and deer, 'e' eyes and the lit window,
       * 'm' gray fur, 'k' dark fur and outlines, 'w' water, 'h' logs:
       */
      static Color role_color(char role)
      {
        const auto &theme = dashboard::theme;
        const auto heat = [&](std::size_t i) {
          return theme.heat[std::min(i, theme.heat.size() - 1)];
        };
        switch (role) {
        case 'f':
          return theme.good;
        case 'l':
          return theme.highlight;
        case 'a':
          return theme.warn;
        case 't':
          return Color::Interpolate(0.4f, theme.muted, theme.track);
        case 'c':
          return theme.bad;
        case 'o':
          return heat(2);
        case 'e':
          return theme.warn;
        case 'm':
          return theme.muted;
        case 'k':
          return theme.track;
        case 'w':
          return Color::Interpolate(0.35f, heat(0), theme.track);
        case 'h':
          return Color::Interpolate(0.45f, heat(2), theme.track);
        default:
          return theme.caption;
        }
      }

      /* A color faded toward the night by @p t from 0 to 1: */
      static Color night(Color c, float t)
      {
        return Color::Interpolate(t, c, dashboard::theme.track);
      }

      /* clang-format off */
      /* A bird of the flock, wings up and down: */
      const std::array<Bitmap, 2> bird_ = {{{"m...m", ".m.m.", "..m.."},
                                            {".....", "mm.mm", "..m.."}}};

      /* The width of a cloud in cells: */
      static constexpr int cloud_width = 24;

      /* The animals, facing right: */
      const std::vector<Bitmap> animal_bitmaps_ = {
          /* a deer */
          {"..........t...t.",
           "..........tt.tt.",
           "...........ttt..",
           "...........oo...",
           "..........oooo..",
           "..........ooeoo.",
           "..........oooooo",
           ".........ooo....",
           ".b......ooo.....",
           ".ooooooooooo....",
           "oooooooooooo....",
           ".oooooooooob....",
           "..o.o.....o.o...",
           "..o.o.....o.o...",
           "..o..o....o..o..",
           "..t..t....t..t.."},
          /* a fox */
          {"............o.o.",
           "............ooo.",
           "...........ooeoo",
           "...........ooooo",
           "oo.........oobbb",
           "ooo.oooooooooob.",
           ".ooooooooooobb..",
           "..bbooooooob....",
           "....o.o...o.o...",
           "....k.k...k.k..."},
          /* a raccoon */
          {"............m.m.",
           "...........mmmmm",
           "..........mkekkm",
           "..........mmmmbb",
           "mkmkm.mmmmmmmm..",
           ".kmkmmmmmmmmm...",
           "..kmmmmmmmmmm...",
           "......m.m..m.m..",
           "......k.k..k.k.."},
          /* a hedgehog */
          {"..t.t.t...",
           ".ttttttt..",
           "tttttttbeb",
           "ttttttbbbk",
           ".b.b..b.b."},
          /* a bunny */
          {"....b.b",
           "....b.b",
           "...bbbb",
           "...bbeb",
           ".bbbbbb",
           "bbbbbb.",
           ".b.b.bb"},
      };

      /* The deer with its head down, grazing: */
      const Bitmap grazing_deer_ = {
           "................",
           "................",
           "................",
           "................",
           "................",
           "................",
           "................",
           "................",
           ".b..............",
           ".oooooooooooo...",
           "oooooooooooooo..",
           ".ooooooooooooo..",
           "..o.o.....o.oot.",
           "..o.o.....o.oeot",
           "..o..o....o..ooo",
           "..t..t....t..t.."};

      /* The owl, looking ahead, left, right, and blinking: */
      const std::array<Bitmap, 4> owl_ = {{
          {".m....m.",
           ".mm..mm.",
           "mmmmmmmm",
           "mbbmmbbm",
           "mbemmebm",
           "mbbmmbbm",
           ".mmbbmm.",
           ".mbbbbm.",
           ".mbbbbm.",
           "..t..t.."},
          {".m....m.",
           ".mm..mm.",
           "mmmmmmmm",
           "bbmmbbmm",
           "ebmmebmm",
           "bbmmbbmm",
           "mmbbmmm.",
           ".mbbbbm.",
           ".mbbbbm.",
           "..t..t.."},
          {".m....m.",
           ".mm..mm.",
           "mmmmmmmm",
           "mmbbmmbb",
           "mmbemmbe",
           "mmbbmmbb",
           ".mmmbbmm",
           ".mbbbbm.",
           ".mbbbbm.",
           "..t..t.."},
          {".m....m.",
           ".mm..mm.",
           "mmmmmmmm",
           "mmmmmmmm",
           "mbbmmbbm",
           "mmmmmmmm",
           ".mmbbmm.",
           ".mbbbbm.",
           ".mbbbbm.",
           "..t..t.."},
      }};

      /* A log cabin with a lit window and a chimney: */
      const Bitmap cabin_bitmap_ = {
          "...............kk.......",
          "...............kk.......",
          "..........ccc..kk.......",
          "........ccccccckk.......",
          "......cccccccccccc......",
          "....cccccccccccccccc....",
          "..cccccccccccccccccccc..",
          "cccccccccccccccccccccccc",
          ".hhhhhhhhhhhhhhhhhhhhhh.",
          ".hkkkkkkhhhhhhhhkkkkkkh.",
          ".hkeekekhhhhhhhhkkkkkkh.",
          ".hkeekekhhhhhhhhkkkkkkh.",
          ".hkkkkkkhhhhhhhhkkkkkkh.",
          ".hkeekekhhhhhhhhkkkkkkh.",
          ".hkeekekhhhhhhhhkkkkkkh.",
          ".hkkkkkkhhhhhhhhkkkkkkh.",
          ".hhhhhhhhhhhhhhhhhhhhhh.",
          ".hhhhhhhhhhhhhhhhhhhhhh."};

      /* A frog, sitting: */
      const Bitmap frog_ = {".ee.ee.",
                            ".fffff.",
                            "fffffff",
                            "f.f.f.f"};

      /* A fern: */
      const Bitmap fern_ = {".fff.",
                            "f.f.f",
                            ".fff.",
                            "..f.."};

      /* A ring of five mushrooms, red caps with light dots: */
      const Bitmap mushrooms_ = {"....c..c..c....",
                                 ".c.cbccbccbc.c.",
                                 "cbc.b..b..b.cbc",
                                 ".b...........b."};

      /* A musical note: */
      const Bitmap note_ = {".xx",
                            ".x.",
                            ".x.",
                            "xx.",
                            "xx."};
      /* clang-format on */

      /*
       * Draw the bitmap @p b into @p into, its bottom row at pixel row @p
       * bottom, in @p color if given, else in the colors of its roles:
       */
      void paint_bitmap(Pixels &into,
                        const Bitmap &b,
                        int x,
                        int bottom,
                        int s,
                        bool left,
                        const Pixels *behind = nullptr,
                        std::optional<Color> color = {}) const
      {
        const int w = static_cast<int>(b.front().size());
        const int h = static_cast<int>(b.size());
        for (int py = 0; py < h * s; ++py)
          for (int px = 0; px < w * s; ++px) {
            const char role = b[py / s][left ? w - 1 - px / s : px / s];
            const Cell c{x + px, bottom - h * s + 1 + py};
            if (role != '.' && (!behind || !behind->contains(c)))
              into.set(c, color ? *color : role_color(role));
          }
      }

      /* The width of a bitmap in cells, and its height in pixels: */
      std::pair<int, int> extent(const Bitmap &b) const
      {
        return {static_cast<int>(b.front().size()) * scale(),
                static_cast<int>(b.size()) * scale()};
      }

      /*
       * Return whether the animal @p kind at column @p x stands on the
       * floor inside the stage, off the pond, and a cell apart from every
       * other animal:
       */
      bool free_floor(int kind, double x, const Animal *self = nullptr) const
      {
        const int w = extent(animal_bitmaps_[kind]).first;
        const int left = nearest(x);
        if (left < 0 || left + w > width_)
          return false;
        if (pond_ && left + w > pond_->first && left < pond_->second)
          return false;
        for (const auto &other : animals_) {
          if (&other == self || !other.visible)
            continue;
          const int other_w = extent(animal_bitmaps_[other.kind]).first;
          const int other_left = nearest(other.x);
          if (left + w + 1 > other_left && left < other_left + other_w + 1)
            return false;
        }
        return true;
      }

      /* A pine with its trunk at column x and a height of h pixels: */
      void pine(Pixels &into,
                int x,
                int h,
                Color needles,
                Color bark,
                Color outline) const
      {
        const int trunk = std::max(2, h / 8);
        const int top = ground() - h + 1;
        const int tier = std::max(4, (h - trunk) / 4);
        /* The widest tier is narrower than the tree is tall: */
        Pixels crown({x - h, x + h, top, top + h - trunk - 1});
        for (int i = 0; i < h - trunk; ++i) {
          /* Tiers that widen downwards: */
          const int half = nearest(0.17 * i + 0.5 * (i % tier));
          for (int dx = -half; dx <= half; ++dx)
            crown.set({x + dx, top + i}, needles);
        }
        paint_trunk(
            into, x, std::max(1, nearest(h / 20.)), ground() - trunk + 1, bark);
        outline_into(into, crown, outline);
      }

      /* Paint a trunk of width w around column x, from row top down: */
      void paint_trunk(Pixels &into, int x, int w, int top, Color bark) const
      {
        for (int py = top; py <= ground(); ++py)
          for (int dx = -(w - 1) / 2; dx <= w / 2; ++dx)
            into.set({x + dx, py}, bark);
      }

      /*
       * A broadleaf tree with its trunk at column x, a height of h pixels,
       * and a round crown of a few overlapping discs:
       */
      void broadleaf(Pixels &into,
                     int x,
                     int h,
                     bool birch,
                     Color leaves,
                     Color bark,
                     Color outline) const
      {
        const int trunk = birch ? h / 2 : h / 3;
        const int w = std::max(1, nearest(h / (birch ? 20. : 12.)));
        paint_trunk(into, x, w, ground() - h + 1, bark);
        if (birch)
          /* Dark dashes on the bark, from the left and right in turn: */
          for (int py = ground() - trunk; py <= ground(); py += 3) {
            const int left = x - (w - 1) / 2;
            const int dash = (w + 1) / 2;
            const int start = (py / 3) % 2 == 0 ? left : left + w - dash;
            for (int px = start; px < start + dash; ++px)
              into.set({px, py}, dashboard::theme.track);
          }

        const double r = (h - trunk) / 2.;
        const double cy = ground() - trunk - r;
        const double reach = birch ? 0.6 : 1.0;
        /* Every disc lies within 2.1 r of the trunk and 1.5 r of cy: */
        const int span = static_cast<int>(std::ceil(3. * r)) + 2;
        Pixels crown(
            {x - span, x + span, nearest(cy) - span, nearest(cy) + span});
        for (int k = 0; k < 5; ++k) {
          /* The discs sit around the center, fixed for every tree: */
          const double ox =
              (k == 0) ? 0. : reach * r * (fraction(x * 5 + k, 50) - 0.5) * 1.6;
          const double oy =
              (k == 0) ? 0. : r * (fraction(x * 5 + k, 51) - 0.6) * 0.8;
          const double rr =
              (k == 0) ? r : r * (0.55 + 0.3 * fraction(x * 5 + k, 52));
          for (int py = nearest(cy + oy - rr); py <= nearest(cy + oy + rr);
               ++py)
            for (int px = nearest(x + ox - rr * reach * 1.3);
                 px <= nearest(x + ox + rr * reach * 1.3);
                 ++px) {
              const double u = (px - x - ox) / (rr * reach * 1.3);
              const double v = (py - cy - oy) / rr;
              if (u * u + v * v <= 1.)
                crown.set({px, py}, leaves);
            }
        }
        outline_into(into, crown, outline);
      }

      /* Draw @p crown into @p into, with its border in @p outline: */
      static void outline_into(Pixels &into, const Pixels &crown, Color outline)
      {
        crown.for_each([&](Cell c, Color color) {
          const bool border = !crown.contains({c.first - 1, c.second}) ||
                              !crown.contains({c.first + 1, c.second}) ||
                              !crown.contains({c.first, c.second - 1}) ||
                              !crown.contains({c.first, c.second + 1});
          into.set(c, border ? outline : color);
        });
      }

      /* Plant one row of trees, each at a fixed column: */
      void tree_row(Pixels &into,
                    int layer,
                    double min_height,
                    double max_height,
                    int min_step,
                    int max_step,
                    float fade)
      {
        for (int x = static_cast<int>(hash(-1, layer) % 6); x < width_ + 12;) {
          const int h =
              std::max(6,
                       nearest(2 * forest_height_ *
                               (min_height + (max_height - min_height) *
                                                 fraction(x, layer))));
          const auto kind = hash(x, layer + 16) % 10;
          const bool birch = kind >= 8;
          const Color leaves = night(role_color(kind < 5 ? 'f'
                                                : birch  ? 'a'
                                                         : 'l'),
                                     fade);
          const Color bark = night(role_color(birch ? 'b' : 't'), fade);
          const Color outline = night(leaves, 0.5f);

          if (kind < 5)
            pine(into, x, h, leaves, bark, outline);
          else
            broadleaf(into, x, h, birch, leaves, bark, outline);

          x += min_step + static_cast<int>(hash(x, layer + 32) %
                                           (max_step - min_step + 1));
        }
      }

      void plant()
      {
        trees_ = Pixels(stage_pixels());
        front_ = Pixels(stage_pixels());
        stars_.clear();
        bushes_.clear();
        owl_perch_.reset();
        cabin_.reset();
        pond_.reset();

        for (int y = 0; y < sky_rows(); ++y)
          for (int x = 0; x < width_; ++x)
            if (hash(x * 977 + y, 48) % 45 == 0)
              stars_.push_back({x, y});

        /* Three rows of trees, fading into the night with distance: */
        tree_row(trees_, 0, 0.85, 1.0, 4, 8, 1.f);
        tree_row(trees_, 1, 0.65, 0.85, 8, 15, 0.65f);
        tree_row(trees_, 2, 0.45, 0.7, 13, 24, 0.4f);

        /*
         * The cabin on the left, about two and a half deer tall, and the
         * pond on the right, if the stage is wide enough:
         */
        const int s = 2 * scale();
        const int cabin_w = static_cast<int>(cabin_bitmap_.front().size()) * s;
        const int cabin_h = static_cast<int>(cabin_bitmap_.size()) * s;
        const int pond_w = 18 * scale();
        const int cabin_x = width_ / 12;
        const int pond_x = width_ * 3 / 4;
        if (cabin_x + cabin_w < pond_x && pond_x + pond_w <= width_) {
          cabin_ = {cabin_x, cabin_x + cabin_w};
          paint_bitmap(trees_, cabin_bitmap_, cabin_x, ground(), s, false);
          chimney_ = {cabin_x + 15 * s, (ground() - cabin_h + 1) / 2 - 1};
          /* The owl on the ridge, bitmap row 2 and columns 10 to 12: */
          owl_perch_ =
              Cell{cabin_x + 10 * s - scale(), ground() - cabin_h + 2 * s};

          pond_ = {pond_x, pond_x + pond_w};
          for (int px = pond_x; px < pond_x + pond_w; ++px) {
            const double u = 2. * (px - pond_x + 0.5) / pond_w - 1.;
            const int depth = nearest(3 * scale() * std::sqrt(1. - u * u));
            for (int py = ground() - depth + 1; py <= ground(); ++py)
              front_.set({px, py}, role_color('w'));
          }
        }

        /* Undergrowth: bushes, some with berries, ferns, a mushroom ring: */
        const auto leaves = night(role_color('l'), 0.35f);
        const auto needles = night(role_color('f'), 0.35f);
        for (int x = static_cast<int>(hash(-1, 3) % 4); x < width_;) {
          const auto h = hash(x, 3);
          const int kind = static_cast<int>(h % 4);
          const int s = scale();
          int w = 0;
          if (pond_ && x + 12 * s > pond_->first && x < pond_->second) {
            x = pond_->second + 1;
            continue;
          }
          if (cabin_ && x + 12 * s > cabin_->first && x < cabin_->second) {
            x = cabin_->second + 1;
            continue;
          }
          if (kind <= 1) {
            /* A bush of two discs, with berries if kind is 1: */
            w = 12 * s;
            /* The larger disc reaches 9 s above the floor: */
            Pixels crown({x, x + w - 1, ground() - 9 * s, ground()});
            for (const auto &[ox, r] :
                 {std::pair{3.5 * s, 3.5 * s}, std::pair{8. * s, 4.5 * s}})
              for (int py = nearest(ground() - r * 2); py <= ground(); ++py)
                for (int px = x; px < x + w; ++px) {
                  const double u = (px - x - ox) / r;
                  const double v = (py - ground() + r) / r;
                  if (u * u + v * v <= 1.)
                    crown.set({px, py}, leaves);
                }
            outline_into(front_, crown, night(leaves, 0.5f));
            if (kind == 1)
              crown.for_each([&](Cell c, Color) {
                if (hash(c.first * 131 + c.second, 6) % 9 == 0)
                  front_.set(c, role_color('c'));
              });
            bushes_.push_back({x, x + w});
          } else if (kind == 2) {
            /* A fern, in the color of the needles: */
            w = static_cast<int>(fern_.front().size()) * s;
            paint_bitmap(
                front_, fern_, x, ground(), s, false, nullptr, needles);
          } else {
            /* A ring of mushrooms, red caps with light dots: */
            w = 14 * s;
            paint_bitmap(front_, mushrooms_, x - s, ground(), s, false);
          }
          x += w + 6 + static_cast<int>((h >> 16) % 16);
        }

        /* Tufts of grass on the floor: */
        for (int x = 0; x < width_; ++x)
          if (!front_.contains({x, ground()}) && hash(x, 5) % 3 != 0) {
            front_.set({x, ground()}, night(needles, 0.3f));
            if (hash(x, 5) % 3 == 1)
              front_.set({x, ground() - 1}, night(needles, 0.3f));
          }
      }

      /* The color of the scenery at pixel @p c, the undergrowth in front: */
      const Color *scenery(Cell c) const
      {
        const auto *color = front_.find(c);
        return color ? color : trees_.find(c);
      }

      /*
       * The glyph of a cell with half blocks, from the colors of its @p top
       * and @p low pixel, of which at most one is null:
       */
      static Glyph cell(const Color *top, const Color *low)
      {
        if (top && low) {
          if (*top == *low)
            return {"█", *top};
          return {"▀", *top, false, false, *low};
        }
        if (top)
          return {"▀", *top};
        return {"▄", *low};
      }

      /* Draw the moon, @p r pixels in radius, waxing over the run: */
      void moon(Pixels &into, int x, int top, int r) const
      {
        /* From a crescent at the start to the full moon at the end: */
        const double phase = 0.25 + 0.75 * progress_;
        const double terminator = r * (1. - 2. * phase);
        for (int py = -r; py < r; ++py)
          for (int px = -r; px < r; ++px) {
            const double u = px + 0.5;
            const double v = py + 0.5;
            if (u * u + v * v > r * r)
              continue;
            const double edge = std::sqrt(r * r - v * v);
            if (u < terminator * edge / r)
              continue;
            const bool crater = hash((px + r) * 31 + py + r, 60) % 7 == 0;
            into.set({x + px + r, top + py + r},
                     crater ? dashboard::theme.muted
                            : dashboard::theme.caption);
          }
      }

      void roam(Animal &a)
      {
        if (!a.visible)
          return;
        if (a.pause > 0) {
          --a.pause;
          a.hiding = a.hiding && a.pause > 0;
          return;
        }

        /* The bunny hides in a bush now and then: */
        if (a.kind == bunny && chance(0.2)) {
          const auto [w, h] = extent(animal_bitmaps_[a.kind]);
          for (const auto &[x0, x1] : bushes_)
            if (nearest(a.x) >= x0 && nearest(a.x) + w <= x1) {
              a.hiding = true;
              a.pause = 30 + pick(60);
              return;
            }
        }
        if (chance(0.01))
          a.pause = 10 + pick(40);
        if (chance(0.005))
          a.right = !a.right;

        /* The bunny hops, the others walk: */
        if (a.kind == bunny && time_ % 4 != 0)
          return;

        /* Turn around at the edges of the stage, the pond, other animals: */
        const double x =
            a.x + (a.right ? 1. : -1.) * (a.kind == bunny ? 1. : 0.25);
        if (free_floor(a.kind, x, &a))
          a.x = x;
        else
          a.right = !a.right;
      }

      void compose(Stage &stage)
      {
        const auto &theme = dashboard::theme;
        Canvas canvas(stage);

        /*
         * The sky in pixels, from back to front: stars, the moon, clouds,
         * birds, shooting stars. It ends where the forest begins:
         */
        Pixels sky_pixels({0, width_ - 1, 0, 2 * sky_rows() - 1});
        for (const auto &[x, y] : stars_) {
          const auto h = hash(x * 977 + y, 49);
          const bool twinkle = (h + time_ / (3 + h % 5)) % 11 == 0;
          const auto color = (h >> 8) % 3 == 0 ? theme.warn : theme.caption;
          sky_pixels.set({x, 2 * y + static_cast<int>((h >> 12) % 2)},
                         twinkle ? color : night(color, 0.4f));
        }

        /* The moon rises on the left and sets on the right over the run: */
        const int r = 6 * scale();
        const int moon_x = nearest(progress_ * std::max(0, width_ - 2 * r));
        const int arc = std::max(0, 2 * sky_rows() - 2 * r - 2);
        const int moon_top =
            1 + nearest(arc * (1. - std::sin(std::numbers::pi *
                                             (0.15 + 0.7 * progress_))));
        moon(sky_pixels, moon_x, moon_top, r);

        /* Soft clouds of a few discs, in front of the stars and the moon: */
        for (const auto &c : clouds_) {
          for (int k = 0; k < 4; ++k) {
            const double ox = 3. + 5. * k;
            const double r = (1.5 + 1.5 * fraction(c.seed * 4 + k, 62)) *
                             (k == 1 || k == 2 ? 1.3 : 1.);
            const double cy = 2 * c.y + 5. - r;
            for (int py = nearest(cy - r); py <= nearest(cy + r); ++py)
              for (int px = nearest(c.x + ox - 1.5 * r);
                   px <= nearest(c.x + ox + 1.5 * r);
                   ++px) {
                const double u = (px - c.x - ox) / (1.5 * r);
                const double v = (py - cy) / r;
                if (u * u + v * v <= 1. && py <= 2 * c.y + 5)
                  sky_pixels.set({px, py}, night(theme.caption, 0.7f));
              }
          }
        }
        if (flock_) {
          /* Five birds in a V, flapping in turn: */
          for (int k = 0; k < 5; ++k) {
            const int back = std::abs(k - 2);
            paint_bitmap(sky_pixels,
                         bird_[(time_ / 3 + k) % 2],
                         flock_->first - 5 * back * scale(),
                         flock_->second + 3 * back * scale(),
                         scale(),
                         false);
          }
        }
        /* A shooting star, a streak that fades toward its tail: */
        for (const auto &m : meteors_)
          for (int k = 0; k <= 3; ++k)
            sky_pixels.set({nearest(m.x - 1.5 * k), nearest(2 * m.y - k)},
                           night(theme.caption, 0.25f * k));
        sky_pixels.for_each_cell([&](int x, int y, auto *top, auto *low) {
          canvas.put(x, y, cell(top, low));
        });

        /* The forest in front of the sky: */
        for (int y = 0; y < height_; ++y)
          for (int x = 0; x < width_; ++x) {
            const auto *top = scenery({x, 2 * y});
            const auto *low = scenery({x, 2 * y + 1});
            if (top || low)
              canvas.put(x, y, cell(top, low));
          }

        /*
         * The figures in pixels, over the scenery: the smoke first, so that
         * the owl stays in front of it, and an animal that hides stays
         * behind the undergrowth:
         */
        Pixels figures(stage_pixels());
        for (const auto &p : smoke_)
          figures.set({nearest(p.x), nearest(2 * p.y)},
                      night(theme.caption, 0.2f + 0.4f * p.age / 40.f));
        if (pond_) {
          /* The moon on the water, and ripples: */
          const int mirror = moon_x + r;
          for (int px = pond_->first + 2; px < pond_->second - 2; ++px)
            if (std::abs(px - mirror) <= scale() ||
                (hash(px * 7 + time_ / 5, 61) % 23 == 0))
              figures.set({px, ground() - 1},
                          std::abs(px - mirror) <= scale()
                              ? theme.caption
                              : night(theme.caption, 0.5f));
          paint_bitmap(figures,
                       frog_,
                       pond_->first - 4 * scale(),
                       ground(),
                       scale(),
                       false);
          /* A note above the frog while it croaks: */
          if (croak_ > 0)
            paint_bitmap(figures,
                         note_,
                         pond_->first - scale(),
                         ground() - 5 * scale(),
                         scale(),
                         false,
                         nullptr,
                         theme.caption);
        }
        if (owl_perch_) {
          const int look = (time_ / 2) % 40 == 0 ? 3
                           : owl_look_ < 0       ? 1
                           : owl_look_ > 0       ? 2
                                                 : 0;
          paint_bitmap(figures,
                       owl_[look],
                       owl_perch_->first,
                       owl_perch_->second,
                       scale(),
                       false);
        }
        for (const auto &a : animals_) {
          if (!a.visible)
            continue;
          const auto &b = (a.kind == deer && a.pause > 0)
                              ? grazing_deer_
                              : animal_bitmaps_[a.kind];
          paint_bitmap(figures,
                       b,
                       nearest(a.x),
                       ground(),
                       scale(),
                       !a.right,
                       a.hiding ? &front_ : nullptr);
        }

        /* Each cell with a figure shows the figure over the scenery: */
        figures.for_each_cell([&](int x, int y, auto *top, auto *low) {
          auto g = cell(top ? top : scenery({x, 2 * y}),
                        low ? low : scenery({x, 2 * y + 1}));
          g.bold = true;
          canvas.put(x, y, std::move(g));
        });

        repaint(stage, frame_, std::move(canvas.cells));
      }

      double progress_ = 0.;
      int width_ = 0;
      int height_ = 0;
      int forest_height_ = 0;
      int time_ = 0;
      int croak_ = 0;
      int owl_look_ = 0;
      std::optional<std::tuple<int, int, dashboard::Palette>> planted_;
      Pixels trees_;
      Pixels front_;
      Frame frame_;
      std::vector<Cell> stars_;
      std::vector<std::pair<int, int>> bushes_;
      std::optional<Cell> owl_perch_;
      std::optional<std::pair<int, int>> cabin_;
      Cell chimney_ = {0, 0};
      std::optional<std::pair<int, int>> pond_;
      std::vector<Cloud> clouds_;
      std::optional<Cell> flock_;
      std::vector<Meteor> meteors_;
      std::vector<Puff> smoke_;
      std::vector<Animal> animals_;
    };
  } // namespace


  std::unique_ptr<Animation> create(DashboardAnimation kind)
  {
    switch (kind) {
    case DashboardAnimation::none:
      return nullptr;
    case DashboardAnimation::ryujin:
      return std::make_unique<Sparkle>();
    case DashboardAnimation::matthias:
      return std::make_unique<Snakes>();
    case DashboardAnimation::eric:
      return std::make_unique<Coffee>();
    case DashboardAnimation::wyatt:
      return std::make_unique<Aquarium>();
    case DashboardAnimation::jordan:
      return std::make_unique<Forest>();
    }

    __builtin_unreachable();
  }
} // namespace ryujin::animation
