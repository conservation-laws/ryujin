//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <driver/dashboard_animation.h>
#include <driver/dashboard_theme.h>
#include <ryujin/base/version_info.h>
#include <ryujin/driver/dashboard.h>

#include <ftxui/dom/canvas.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/dom/node.hpp>
#include <ftxui/screen/screen.hpp>
#include <ftxui/screen/terminal.hpp>

#include <sys/ioctl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <format>
#include <iostream>
#include <numbers>
#include <random>
#include <ranges>

namespace ryujin
{
  using namespace ftxui;
  using namespace dashboard;

  namespace
  {
    /* Return the escape sequences that redraw the changed cells: */
    std::string changes(animation::Stage &stage)
    {
      auto &changed = stage.changed;
      std::ranges::sort(changed);
      changed.erase(std::ranges::unique(changed).begin(), changed.end());

      std::string output;
      for (const auto &[x, y] : changed) {
        const auto &pixel = stage.screen.PixelAt(x, y);
        output += std::format("\033[{};{}H\033[0;", y + 1, x + 1) +
                  (pixel.bold ? "1;" : "") + (pixel.dim ? "2;" : "") +
                  pixel.foreground_color.Print(false) + ";" +
                  pixel.background_color.Print(true) + "m" +
                  (pixel.character.empty() ? " " : pixel.character);
      }

      /* Save and restore the cursor position (DECSC and DECRC): */
      return output.empty() ? "" : "\0337" + output + "\033[0m\0338";
    }


    /*
     * Formatting helpers:
     */

    /* Map x in [0, 1] onto the heat scale of the palette: */
    Color heat(double x)
    {
      const auto &stops = theme.heat;
      x = std::clamp(x, 0., 1.) * static_cast<double>(stops.size() - 1);
      const auto i = std::min(static_cast<std::size_t>(x), stops.size() - 2);
      const auto t = static_cast<float>(x - static_cast<double>(i));
      return Color::Interpolate(t, stops[i], stops[i + 1]);
    }


    std::string format_duration(double seconds)
    {
      if (!std::isfinite(seconds) || seconds < 0.)
        return "--";

      auto s = static_cast<unsigned long>(seconds);
      const auto days = s / 86400;
      const auto hours = (s % 86400) / 3600;
      const auto minutes = (s % 3600) / 60;
      s %= 60;

      if (days > 0)
        return std::format("{}d {:02}h {:02}m", days, hours, minutes);
      if (hours > 0)
        return std::format("{}h {:02}m {:02}s", hours, minutes, s);
      return std::format("{}m {:02}s", minutes, s);
    }


    std::string format_si(double value)
    {
      for (const auto [unit, prefix] :
           {std::pair{1.e9, 'G'}, {1.e6, 'M'}, {1.e3, 'k'}})
        if (value >= unit)
          return std::format("{:.2f} {}", value / unit, prefix);
      return std::format("{:.0f} ", value);
    }


    /* Round up to the next number of the form {1, 1.5, 2, ..., 8} * 10^k: */
    double nice_scale(double value)
    {
      if (!(value > 0.) || !std::isfinite(value))
        return 1.;
      const double magnitude = std::pow(10., std::floor(std::log10(value)));
      for (const double step : {1., 1.5, 2., 2.5, 3., 4., 5., 6., 8.})
        if (step * magnitude >= value)
          return step * magnitude;
      return 10. * magnitude;
    }


    double safe(double value)
    {
      return std::isfinite(value) ? value : 0.;
    }


    /*
     * Elements:
     */

    /*
     * Limit the natural height of an element to `height` rows, but let it
     * grow into all the space that is available:
     */
    class AtMost : public Node
    {
    public:
      AtMost(Element child, int height)
          : Node({std::move(child)})
          , height_(height)
      {
      }

      void ComputeRequirement() override
      {
        Node::ComputeRequirement();
        requirement_.min_y = std::min(requirement_.min_y, height_);
        requirement_.flex_grow_y = 1;
        requirement_.flex_shrink_y = 1;
      }

      void SetBox(Box box) override
      {
        Node::SetBox(box);
        children_[0]->SetBox(box);
      }

    private:
      int height_;
    };


    /* A gauge of @p fraction in [0, 1]: */
    Element bar(double fraction, Decorator style)
    {
      return gauge(static_cast<float>(std::clamp(safe(fraction), 0., 1.))) |
             style | bgcolor(theme.track);
    }


    /* A labeled gauge of @p fraction, the percentage, and @p details: */
    Element meter(const std::string &name,
                  double fraction,
                  Decorator style,
                  const std::string &details)
    {
      return hbox({
          label(name),
          bar(fraction, std::move(style)) | size(WIDTH, EQUAL, 24),
          text(std::format(" {:5.1f}%", 100. * fraction)) | bold,
          muted(details),
      });
    }


    /* A sparkline of the most recent `width` values of a history: */
    Element sparkline(const std::deque<double> &history, int width, Color c)
    {
      static const char *const blocks[] = {
          "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};

      const auto n = std::min<int>(width, history.size());
      const auto [min, max] =
          std::ranges::minmax(std::views::drop(history, history.size() - n));
      const double range = std::max(max - min, 1.e-3 * max);

      Elements elements = {text(std::string(width - n, ' '))};
      for (auto it = history.end() - n; it != history.end(); ++it) {
        const double level =
            range > 0. ? std::clamp((*it - min) / range, 0., 1.) : 0.5;
        elements.push_back(
            text(blocks[static_cast<int>(std::round(level * 7.))]) |
            color(Color::Interpolate(float(level), theme.muted, c)));
      }
      return hbox(std::move(elements));
    }


    /* A range bar marking [min, max] and the average relative to scale: */
    Element
    range_bar(const DashboardData::MinMaxAvg &v, double scale, int width)
    {
      const auto position = [&](double value) {
        const auto column = static_cast<int>(value / scale * (width - 1));
        return std::clamp(column, 0, width - 1);
      };

      Elements elements;
      for (int i = 0; i < width; ++i) {
        if (i == position(v.avg))
          elements.push_back(text("◆") | color(theme.warn));
        else if (i < position(v.min))
          elements.push_back(text("━") | color(theme.muted));
        else if (i <= position(v.max))
          elements.push_back(text("━") | color(theme.accent));
        else
          elements.push_back(text("─") | color(theme.track));
      }
      return hbox(std::move(elements));
    }


    /* A speedometer with a needle pointing at value in [0, scale]: */
    Element speedometer(double value, double scale)
    {
      constexpr int cx = 29;
      constexpr int cy = 27;
      constexpr int radius = 25;

      const double fraction = std::clamp(safe(value) / scale, 0., 1.);

      Canvas c(58, 28);

      const auto point = [&](double s, double r) {
        const double phi = std::numbers::pi * (1. - s);
        return std::pair{static_cast<int>(std::round(cx + r * std::cos(phi))),
                         static_cast<int>(std::round(cy - r * std::sin(phi)))};
      };

      /* The dial: */
      for (int i = 0; i <= 160; ++i) {
        const double s = i / 160.;
        const auto dial =
            s <= fraction
                ? Color::Interpolate(float(s), theme.highlight, theme.good)
                : theme.muted;
        for (const double r : {radius, radius - 1, radius - 2}) {
          const auto [x, y] = point(s, r);
          c.DrawPoint(x, y, true, dial);
        }
      }

      /* Ticks: */
      for (int i = 0; i <= 10; ++i) {
        const bool major = i % 5 == 0;
        const auto [x1, y1] = point(i / 10., radius - 5);
        const auto [x2, y2] = point(i / 10., major ? radius - 9 : radius - 7);
        c.DrawPointLine(x1, y1, x2, y2, major ? theme.caption : theme.muted);
      }

      /* The needle: */
      const auto [x, y] = point(fraction, radius - 8);
      c.DrawPointLine(cx, cy, x, y, theme.warn);
      c.DrawPointLine(cx + 1, cy, x + 1, y, theme.warn);
      c.DrawPointCircleFilled(cx, cy, 2, theme.bad);

      return canvas(std::move(c));
    }


    /* A seven-segment display of value with four significant digits: */
    Element digital(double value)
    {
      constexpr int w = 10;
      constexpr int h = 24;
      constexpr int y0 = 2;

      /* The segments a to g as rectangles {x1, y1, x2, y2}: */
      constexpr std::array<std::array<int, 4>, 7> segments = {{
          {0, 0, w - 1, 1},
          {w - 2, 0, w - 1, h / 2},
          {w - 2, h / 2 - 1, w - 1, h - 1},
          {0, h - 2, w - 1, h - 1},
          {0, h / 2 - 1, 1, h - 1},
          {0, 0, 1, h / 2},
          {0, h / 2 - 1, w - 1, h / 2},
      }};

      /* The lit segments of the digits 0 to 9, one bit per segment: */
      constexpr std::array<int, 10> digits = {
          0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F};

      const double v = safe(value);
      const int precision = v < 10. ? 3 : v < 100. ? 2 : v < 1000. ? 1 : 0;
      auto string = std::format("{:.{}f}", v, precision);

      /* Drop a decimal if rounding carries over into another digit: */
      if (precision > 0 && string.size() > 5)
        string = std::format("{:.{}f}", v, precision - 1);

      const auto advance = [](char ch) { return ch == '.' ? 4 : w + 4; };
      int width = -4;
      for (const char ch : string)
        width += advance(ch);
      Canvas c(width, 28);

      /* Fill the rectangle {x1, y1, x2, y2} at the offset (x0, y0): */
      int x0 = 0;
      const auto fill = [&](int x1, int y1, int x2, int y2) {
        for (int y = y1; y <= y2; ++y)
          c.DrawPointLine(x0 + x1, y0 + y, x0 + x2, y0 + y, theme.good);
      };

      for (const char ch : string) {
        if (ch == '.')
          fill(0, h - 2, 1, h - 1);
        else
          for (int i = 0; i < 7; ++i)
            if ((digits[ch - '0'] >> i & 1) != 0)
              std::apply(fill, segments[i]);
        x0 += advance(ch);
      }

      return canvas(std::move(c));
    }


    /* The restarts, warnings, or corrections of the two modules: */
    Element counter(const std::string &name, std::array<unsigned int, 2> n)
    {
      const auto c = [](unsigned int k) {
        if (k == 0)
          return muted("0");
        return text(std::to_string(k)) | bold |
               color(k < 10 ? theme.warn : theme.bad);
      };
      return hbox(
          {caption(name + " "), c(n[0]), muted("/"), c(n[1]), text("   ")});
    }


    /* The dashboard for the snapshot @p data, @p width columns wide: */
    Element document(const DashboardData &data,
                     const std::deque<double> &wall_history,
                     const std::deque<double> &compute_history,
                     DashboardThroughputDisplay display,
                     int width)
    {
      /*
       * Header: cycle, time, and progress
       */

      /* The last time step may overshoot the final time: */
      const double progress =
          data.t_final > 0. ? std::clamp(data.t / data.t_final, 0., 1.) : 0.;

      auto header = vbox({
          hbox({
              /* The version is also shown in the footer: */
              badge(width >= 100),
              text(" "),
              data.final_time ? text(" FINAL ") | bold | color(theme.ink) |
                                    bgcolor(theme.good)
                              : text(" running ") | color(theme.ink) |
                                    bgcolor(theme.highlight),
              caption("   cycle "),
              text(std::format("{}", data.cycle)) | bold,
              caption("   t = "),
              text(std::format("{:.8f}", data.t)) | bold,
              muted(std::format(" / {:.4g}", data.t_final)),
              filler(),
              caption("ETA "),
              text(format_duration(data.eta)) | bold,
              text(" "),
          }),
          hbox({
              bar(progress,
                  color(LinearGradient(theme.highlight, theme.good))) |
                  flex,
              text(std::format(" {:5.1f}% ", 100. * progress)) | bold,
          }),
      });

      /*
       * Information
       */

      Elements information = {
          hbox({label("hyperbolic"), text(data.hyperbolic_name) | bold}),
      };
      if (!data.parabolic_name.empty())
        information.push_back(
            hbox({label("parabolic"), text(data.parabolic_name) | bold}));

      std::string ranks =
          std::format("{} rank{}", data.n_ranks, data.n_ranks == 1 ? "" : "s");
      if (!data.threads.empty())
        ranks += " × " + data.threads;
      if (data.n_ensembles > 1)
        ranks += std::format(" ({} ensembles)", data.n_ensembles);

      information.push_back(hbox({label("run"), text(data.base_name)}));
      information.push_back(hbox({
          label("size"),
          text(format_si(data.n_global_dofs)) | bold,
          text("Qdofs"),
      }));
      information.push_back(hbox({
          label("parallel"),
          text(ranks),
          filler(),
          caption("backend "),
          chip(data.backend, theme.highlight),
      }));

      /*
       * Throughput
       */

      /* Scale with the recent samples to forget the warm-up phase: */
      const auto recent = std::min<std::size_t>(wall_history.size(), 32);
      const double scale =
          nice_scale(1.1 * *std::max_element(wall_history.end() - recent,
                                             wall_history.end()));

      const bool dial = display == DashboardThroughputDisplay::speedometer;

      auto throughput = vbox({
          hbox({
              filler(),
              dial ? speedometer(data.wall_m_dofs_per_sec, scale)
                   : digital(data.wall_m_dofs_per_sec),
              filler(),
          }),
          hbox({
              muted(dial ? " 0" : ""),
              filler(),
              text(std::format("{:.2f}", data.wall_m_dofs_per_sec)) | bold |
                  color(theme.good),
              text(" MQ/s wall"),
              filler(),
              muted(dial ? std::format("{:g} ", scale) : ""),
          }),
          separatorLight() | color(theme.track),
          hbox({
              caption(data.device               ? "GPU "
                      : data.fudged_cpu_timings ? "CPU*"
                                                : "CPU "),
              text(std::format(" {:9.3f} MQ/s ", data.compute_m_dofs_per_sec)),
              filler(),
              sparkline(compute_history, 14, theme.highlight),
          }),
          hbox({
              caption("WALL"),
              text(std::format(" {:9.3f} MQ/s ", data.wall_m_dofs_per_sec)),
              filler(),
              sparkline(wall_history, 14, theme.good),
          }),
          hbox({
              caption("rate"),
              text(std::format(" {:9.2f} cycles/s", data.cycles_per_second)),
          }),
      });

      /*
       * Throughput history: an area chart of the most recent wall
       * throughput samples, one sample per half cell
       */

      auto history_graph = [&wall_history, scale](int width, int height) {
        /* A negative value leaves the column empty: */
        std::vector<int> values(width, -1);
        const auto n = std::min<std::size_t>(width, wall_history.size());
        for (std::size_t i = 0; i < n; ++i) {
          const double value = wall_history[wall_history.size() - n + i];
          const auto level =
              static_cast<int>(std::round(value / scale * height));
          values[width - n + i] = std::clamp(level, 0, height);
        }
        return values;
      };

      auto history = hbox({
          vbox({muted(std::format("{:g}", scale)), filler(), muted("0")}) |
              size(WIDTH, EQUAL, 5),
          graph(std::move(history_graph)) | color(theme.good) | flex,
      });

      /*
       * Messages: the most recent status messages, scrolled to the newest
       */

      Elements message_lines;
      for (const auto &message : data.messages)
        message_lines.push_back(hbox({
                                    muted(std::format("{:>6} ", message.cycle)),
                                    text(message.text),
                                }) |
                                flex_shrink);
      if (message_lines.empty())
        message_lines.push_back(muted("no messages"));

      auto messages =
          std::make_shared<AtMost>(vbox(std::move(message_lines)) |
                                       focusPositionRelative(0.f, 1.f) | yframe,
                                   5);

      /*
       * Time stepping
       */

      Elements stepping = {
          hbox({
              label("scheme"),
              chip(data.time_stepping_scheme, theme.good),
              caption("  CFL "),
              text(std::format("{:.2f}", data.cfl)) | bold,
          }),
          hbox({
              label("tau"),
              text(std::format("{:.3e}", data.tau)) | bold,
              muted(std::format("   ({:.3e} t/s)", data.time_per_second)),
          }),
          hbox({
              label("events"),
              counter("restarts", data.restarts),
              counter("warnings", data.warnings),
              counter("corrections", data.corrections),
          }),
      };

      for (const auto &line : data.solver_statistics)
        stepping.push_back(hbox({label("solver"), text(line)}));

      if (data.device) {
        stepping.push_back(meter("GPU util",
                                 data.utilization,
                                 color(LinearGradient(theme.bad, theme.good)),
                                 std::format("  on {} device{}",
                                             data.n_devices,
                                             data.n_devices == 1 ? "" : "s")));
        const auto skew =
            std::clamp(safe(data.device_time_skew_fraction), 0., 1.);
        stepping.push_back(meter("gpu skew",
                                 skew,
                                 color(heat(4. * skew)),
                                 std::format("  ({:.2e} s/cycle) [p{}/{}]",
                                             data.device_time_skew,
                                             data.device_time_min_index,
                                             data.device_time_max_index)));
      } else {
        const auto skew = std::clamp(safe(data.cpu_time_skew_fraction), 0., 1.);
        stepping.push_back(
            meter("cpu skew",
                  skew,
                  color(heat(4. * skew)),
                  std::format("  ({:.2e} s/cycle)", data.cpu_time_skew)));
      }

      /*
       * Memory
       */

      const auto &memory = data.memory;
      auto memory_panel = vbox({
          hbox({
              caption("min "),
              text(std::format("{:9.1f}", memory.min)),
              muted(std::format(" [p{}]", memory.min_index)),
          }),
          hbox({caption("avg "),
                text(std::format("{:9.1f}", memory.avg)) | bold}),
          hbox({
              caption("max "),
              text(std::format("{:9.1f}", memory.max)),
              muted(std::format(" [p{}]", memory.max_index)),
          }),
          range_bar(memory, nice_scale(memory.max), 32),
          hbox({
              muted("0"),
              filler(),
              muted(std::format("{:g} MiB", nice_scale(memory.max))),
          }),
      });

      /*
       * Timers: a stacked colorbar of the time loop and a table
       */

      const int composition_width = width - 4;
      Elements composition;
      std::vector<Elements> timer_table = {{
          text(" "),
          text("timer") | bold | flex_shrink,
          text("wall avg ") | bold | align_right,
          text(" skew -/+ ") | bold | align_right,
          text(" ranks ") | bold,
          text("cpu sum ") | bold | align_right,
          text("share") | bold,
          text(""),
      }};

      int used = 0;
      std::size_t n_segment = 0;
      for (const auto &timer : data.timers) {
        const auto &wall = timer.wall_time;
        const auto skew = [&wall](double value) {
          if (!(wall.avg > 0.))
            return 0.;
          return std::clamp(100. * (value / wall.avg - 1.), -99.9, 99.9);
        };
        const double skew_negative = skew(wall.min);
        const double skew_positive = skew(wall.max);

        Element marker = text(" ");
        Element share = text("");
        Element percentage = text("");

        if (timer.fraction >= 0.) {
          const auto c = theme.segments[n_segment++ % theme.segments.size()];
          const auto fraction = std::clamp(safe(timer.fraction), 0., 1.);
          const int n = std::min(
              static_cast<int>(std::round(fraction * composition_width)),
              composition_width - used);
          if (n > 0) {
            composition.push_back(text(std::string(n, ' ')) | bgcolor(c));
            used += n;
          }
          marker = text("■ ") | color(c);
          share = bar(fraction, color(c)) | size(WIDTH, EQUAL, 20);
          percentage = text(std::format(" {:5.1f}%", 100. * fraction));
        }

        /* Every timer of the time loop is a "time step": */
        std::string name = timer.name;
        if (name.starts_with("time step "))
          name.erase(0, 10);

        timer_table.push_back({
            marker,
            /* Truncate the name first if the terminal is too narrow: */
            text(name + "  ") | flex_shrink,
            text(std::format("{:.2f}s ", wall.avg)) | align_right,
            text(std::format(
                " {:5.1f}/{:4.1f}% ", skew_negative, skew_positive)) |
                color(heat(std::max(-skew_negative, skew_positive) / 25.)) |
                align_right,
            muted(std::format(" p{}/p{} ", wall.min_index, wall.max_index)),
            text(std::format("{:.2f}s ", timer.cpu_time)) | align_right,
            share,
            percentage,
        });
      }

      composition.push_back(text(std::string(composition_width - used, ' ')) |
                            bgcolor(theme.track));

      /*
       * Tasks: the trigger of every task, when it fired last, and when it
       * fires next
       */

      std::vector<Elements> task_table = {{
          text("task") | bold,
          text("  phase") | bold,
          text("  trigger") | bold,
          text("  last fired") | bold | flex_shrink,
          text("  ") | bold,
          text(" next") | bold,
      }};

      for (const auto &task : data.tasks) {
        task_table.push_back({
            text(task.name) | bold,
            hbox({text("  "), chip(task.phase, theme.highlight)}),
            task.trigger.empty() ? muted("  --") : caption("  " + task.trigger),
            (task.last.empty() ? muted("  --") : text("  " + task.last)) |
                flex_shrink,
            hbox({
                text("  "),
                task.progress < 0.
                    ? text("")
                    : bar(task.progress, color(theme.highlight)) |
                          size(WIDTH, EQUAL, 16),
            }),
            task.next.empty() ? muted(" --")
                              : text(" " + task.next) | color(theme.good),
        });
      }

      /*
       * Assemble the dashboard:
       */

      return vbox({
          borderRounded(header | color(Color::Default)) | color(theme.accent),
          hbox({
              vbox({
                  panel("simulation", vbox(std::move(information))),
                  hbox({
                      panel("wall throughput history [MQ/s]", history) | flex,
                      panel("messages", messages) | flex,
                  }) | yflex,
                  panel("time stepping", vbox(std::move(stepping))),
              }) | flex,
              vbox({
                  panel("throughput", throughput),
                  panel("memory [MiB]", memory_panel) | yflex,
              }) | size(WIDTH, EQUAL, 36),
          }),
          panel("timers",
                vbox({
                    hbox(std::move(composition)),
                    text(""),
                    gridbox(std::move(timer_table)),
                })),
          panel("tasks", gridbox(std::move(task_table))),
          hbox({
              muted(" terminal update every "),
              caption(std::format("{:.2f}s", data.terminal_update_interval)),
              filler(),
              muted(version_string() + " "),
          }),
      });
    }
  } // namespace


  std::pair<int, int> DashboardPosition::offset(DashboardLocation location,
                                                int free_columns,
                                                int free_rows) const
  {
    const auto [x, y] =
        location == DashboardLocation::top_left ? std::pair{0., 0.}
        : location == DashboardLocation::center ? std::pair{.5, .5}
                                                : fractions_;
    return {static_cast<int>(x * free_columns),
            static_cast<int>(y * free_rows)};
  }


  bool DashboardPosition::move(DashboardLocation location)
  {
    if (location != DashboardLocation::screen_saver ||
        std::chrono::steady_clock::now() < next_move_)
      return false;

    static std::mt19937 random{std::random_device{}()};
    std::uniform_real_distribution<double> uniform;
    fractions_ = {uniform(random), uniform(random)};
    postpone();
    return true;
  }


  void DashboardPosition::postpone()
  {
    next_move_ = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  }


  Dashboard::Dashboard() = default;


  Dashboard::~Dashboard()
  {
    /* Stop the background thread: */
    thread_ = {};
    if (terminal_ != nullptr)
      std::fclose(terminal_);
  }


  void Dashboard::start(const DashboardSettings &settings)
  {
    std::scoped_lock lock(mutex_);

    terminal_ = std::fopen("/dev/tty", "we");
    settings_ = settings;
    theme = make_palette(settings.color_scheme);
    animation_ = animation::create(settings.animation);
    write(render());

    thread_ = std::jthread([this](std::stop_token token) {
      std::condition_variable_any wakeup;
      std::unique_lock lock(mutex_);
      while (!wakeup.wait_for(
          lock, token, std::chrono::milliseconds(150), [&token] {
            return token.stop_requested();
          })) {
        /*
         * Redraw the last frame right away if the terminal is resized, or
         * if the dashboard moves. Otherwise advance the animation, unless
         * the last frame has scrolled:
         */
        const auto size = terminal_size();
        if (size != terminal_size_ ||
            (data_ && position_.move(settings_.location))) {
          write(render());
        } else if (animation_ && stage_->screen.dimy() <= size.second) {
          stage_->changed.clear();
          animation_->step(*stage_);
          write(changes(*stage_));
        }
      }
    });
  }


  void Dashboard::show(const DashboardData &data)
  {
    /* No more updates after the final snapshot: */
    if (data.final_time)
      thread_ = {};

    std::scoped_lock lock(mutex_);

    wall_history_.push_back(safe(data.wall_m_dofs_per_sec));
    compute_history_.push_back(safe(data.compute_m_dofs_per_sec));
    if (wall_history_.size() > 512) {
      wall_history_.pop_front();
      compute_history_.pop_front();
    }

    data_ = data;
    position_.move(settings_.location);

    /* Continue with subsequent output below the final dashboard: */
    write(render() + (data.final_time ? "\n" : ""));
  }


  std::string Dashboard::render()
  {
    const auto [columns, lines] = terminal_size();
    const bool final_time = data_ && data_->final_time;

    /* Clear the screen if the terminal has been resized, and at the end: */
    const bool clear =
        std::pair{columns, lines} != terminal_size_ || final_time;
    terminal_size_ = {columns, lines};

    if (!data_) {
      /* A stage without dashboard: */
      stage_ = std::make_unique<animation::Stage>(Screen(columns, lines),
                                                  Box{0, -1, 0, -1});
    } else {
      /* Limit the width of the dashboard on wide terminals: */
      const int terminal_columns = std::max(columns, 80);
      const int width =
          settings_.width == 0
              ? terminal_columns
              : std::min(terminal_columns, std::max<int>(settings_.width, 80));
      auto element = document(*data_,
                              wall_history_,
                              compute_history_,
                              settings_.throughput_display,
                              width) |
                     size(WIDTH, EQUAL, width);

      /* The natural height, stretched up to the maximal height: */
      const int height = std::max(Dimension::Fit(element, true).dimy,
                                  std::min<int>(settings_.height, lines));

      /* Cover the whole terminal, or more if the dashboard does not fit: */
      const int rows = std::max(height, lines);
      const auto [left, top] = position_.offset(
          settings_.location, terminal_columns - width, rows - height);

      stage_ = std::make_unique<animation::Stage>(
          Screen(terminal_columns, rows),
          Box{left, left + width - 1, top, top + height - 1});
      Render(stage_->screen,
             place(element | size(HEIGHT, EQUAL, height), left, top));
    }

    /* Leave the screen around the final dashboard empty: */
    if (animation_ && !final_time)
      animation_->draw(*stage_);

    std::string output = clear ? "\033[2J\033[H" : "\033[H";
    output += stage_->screen.ToString();

    /* If the dashboard does not fit, clear whatever is left below: */
    if (stage_->screen.dimy() > lines)
      output += "\n\033[J";

    return output;
  }


  std::pair<int, int> Dashboard::terminal_size() const
  {
    winsize size{};
    if (terminal_ != nullptr &&
        ::ioctl(fileno(terminal_), TIOCGWINSZ, &size) == 0 && size.ws_col > 0 &&
        size.ws_row > 0)
      return {size.ws_col, size.ws_row};

    const auto fallback = Terminal::Size();
    return {fallback.dimx, fallback.dimy};
  }


  void Dashboard::write(const std::string &output) const
  {
    /* Write out everything that is still pending on stdout first: */
    std::cout << std::flush;

    const auto stream = terminal_ != nullptr ? terminal_ : stdout;
    std::fwrite(output.data(), 1, output.size(), stream);
    std::fflush(stream);
  }
} // namespace ryujin
