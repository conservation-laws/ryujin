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

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <numbers>
#include <random>

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
      if (value >= 1.e9)
        return std::format("{:.2f} G", value / 1.e9);
      if (value >= 1.e6)
        return std::format("{:.2f} M", value / 1.e6);
      if (value >= 1.e3)
        return std::format("{:.2f} k", value / 1.e3);
      return std::format("{:.0f} ", value);
    }


    /* Round up to the next number of the form {1, 1.5, 2, ..., 8} * 10^k: */
    double nice_scale(double value)
    {
      if (!(value > 0.) || !std::isfinite(value))
        return 1.;
      const double magnitude = std::pow(10., std::floor(std::log10(value)));
      for (const double step : {1., 1.5, 2., 2.5, 3., 4., 5., 6., 8., 10.})
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


    Decorator at_most(int height)
    {
      return [height](Element child) {
        return std::make_shared<AtMost>(std::move(child), height);
      };
    }


    Element bar(double fraction, Color begin, Color end)
    {
      return gauge(static_cast<float>(std::clamp(safe(fraction), 0., 1.))) |
             color(LinearGradient(begin, end)) | bgcolor(theme.track);
    }


    /* A sparkline of the most recent `width` values of a history: */
    Element sparkline(const std::deque<double> &history, int width, Color c)
    {
      static const char *const blocks[] = {
          "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};

      const auto n = std::min<int>(width, history.size());
      const auto [min_it, max_it] =
          std::minmax_element(history.end() - n, history.end());
      const double min = *min_it;
      const double range = std::max(*max_it - min, 1.e-3 * *max_it);

      Elements elements;
      if (n < width)
        elements.push_back(text(std::string(width - n, ' ')));

      for (auto it = history.end() - n; it != history.end(); ++it) {
        const double level =
            range > 0. ? std::clamp((safe(*it) - min) / range, 0., 1.) : 0.5;
        const auto index =
            std::clamp(static_cast<int>(std::round(level * 7.)), 0, 7);
        const auto t = static_cast<float>(level);
        elements.push_back(text(blocks[index]) |
                           color(Color::Interpolate(t, theme.muted, c)));
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

      const auto min = position(v.min);
      const auto max = position(v.max);
      const auto avg = position(v.avg);

      Elements elements;
      for (int i = 0; i < width; ++i) {
        if (i == avg)
          elements.push_back(text("◆") | color(theme.warn));
        else if (i >= min && i <= max)
          elements.push_back(text("━") | color(theme.accent));
        else if (i < min)
          elements.push_back(text("━") | color(theme.muted));
        else
          elements.push_back(text("─") | color(theme.track));
      }
      return hbox(std::move(elements));
    }


    /* A speedometer with a needle pointing at value in [0, scale]: */
    Element speedometer(double value, double scale)
    {
      constexpr int width = 58;
      constexpr int height = 28;
      constexpr int cx = 29;
      constexpr int cy = 27;
      constexpr int radius = 25;
      constexpr double pi = std::numbers::pi;

      const double fraction = std::clamp(safe(value) / scale, 0., 1.);

      Canvas c(width, height);

      const auto point = [&](double s, double r) {
        const double phi = pi * (1. - s);
        return std::pair{static_cast<int>(std::round(cx + r * std::cos(phi))),
                         static_cast<int>(std::round(cy - r * std::sin(phi)))};
      };

      /* The dial: */
      constexpr int n_steps = 160;
      for (int i = 0; i <= n_steps; ++i) {
        const double s = static_cast<double>(i) / n_steps;
        const auto t = static_cast<float>(s);
        const auto dial =
            s <= fraction ? Color::Interpolate(t, theme.highlight, theme.good)
                          : theme.muted;
        for (const double r : {radius, radius - 1, radius - 2}) {
          const auto [x, y] = point(s, r);
          c.DrawPoint(x, y, true, dial);
        }
      }

      /* Ticks: */
      for (int i = 0; i <= 10; ++i) {
        const double s = i / 10.;
        const auto [x1, y1] = point(s, radius - 5);
        const auto [x2, y2] = point(s, i % 5 == 0 ? radius - 9 : radius - 7);
        c.DrawPointLine(
            x1, y1, x2, y2, i % 5 == 0 ? theme.caption : theme.muted);
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

      int width = 0;
      for (const char ch : string)
        width += ch == '.' ? 4 : w + 4;
      Canvas c(width - 4, 28);

      int x0 = 0;
      for (const char ch : string) {
        if (ch == '.') {
          c.DrawPointLine(x0, y0 + h - 2, x0 + 1, y0 + h - 2, theme.good);
          c.DrawPointLine(x0, y0 + h - 1, x0 + 1, y0 + h - 1, theme.good);
          x0 += 4;
          continue;
        }

        for (int i = 0; i < 7; ++i) {
          if ((digits[ch - '0'] >> i & 1) == 0)
            continue;
          const auto [x1, y1, x2, y2] = segments[i];
          for (int y = y1; y <= y2; ++y)
            c.DrawPointLine(x0 + x1, y0 + y, x0 + x2, y0 + y, theme.good);
        }
        x0 += w + 4;
      }

      return canvas(std::move(c));
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


  std::string Dashboard::render(const DashboardData &data,
                                const DashboardSettings &settings)
  {
    theme = make_palette(settings.color_scheme);

    /* Check whether the terminal has been resized: */
    const auto [columns, lines] = terminal_size();
    const bool resized =
        columns != terminal_width_ || lines != terminal_height_;
    terminal_width_ = columns;
    terminal_height_ = lines;

    /* Limit the width of the dashboard on wide terminals: */
    const int terminal_columns = std::max(terminal_width_, 80);
    const int screen_width =
        settings.width == 0
            ? terminal_columns
            : std::min(terminal_columns, std::max<int>(settings.width, 80));

    /*
     * Header: cycle, time, and progress
     */

    /* The last time step may overshoot the final time: */
    const double progress =
        data.t_final > 0. ? std::clamp(data.t / data.t_final, 0., 1.) : 0.;

    auto header = vbox({
        hbox({
            /* The version is also shown in the footer: */
            badge(screen_width >= 100),
            text(" "),
            data.final_time ? text(" FINAL ") | bold | color(theme.ink) |
                                  bgcolor(theme.good)
                            : text(" running ") | color(theme.ink) |
                                  bgcolor(theme.highlight),
            text("   cycle ") | color(theme.caption),
            text(std::format("{}", data.cycle)) | bold,
            text("   t = ") | color(theme.caption),
            text(std::format("{:.8f}", data.t)) | bold,
            text(std::format(" / {:.4g}", data.t_final)) | color(theme.muted),
            filler(),
            text("ETA ") | color(theme.caption),
            text(format_duration(data.eta)) | bold,
            text(" "),
        }),
        hbox({
            bar(progress, theme.highlight, theme.good) | flex,
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
        text("backend ") | color(theme.caption),
        chip(data.backend, theme.highlight),
    }));

    /*
     * Throughput speedometer
     */

    /* Scale with the recent samples to forget the warm-up phase: */
    const auto recent = std::min<std::size_t>(wall_history_.size(), 32);
    const double scale =
        nice_scale(1.1 * *std::max_element(wall_history_.end() - recent,
                                           wall_history_.end()));

    const std::string compute_label = data.device               ? "GPU "
                                      : data.fudged_cpu_timings ? "CPU*"
                                                                : "CPU ";

    const bool dial =
        settings.throughput_display == DashboardThroughputDisplay::speedometer;

    auto throughput = vbox({
        hbox({
            filler(),
            dial ? speedometer(data.wall_m_dofs_per_sec, scale)
                 : digital(data.wall_m_dofs_per_sec),
            filler(),
        }),
        hbox({
            text(dial ? " 0" : "") | color(theme.muted),
            filler(),
            text(std::format("{:.2f}", data.wall_m_dofs_per_sec)) | bold |
                color(theme.good),
            text(" MQ/s wall"),
            filler(),
            text(dial ? std::format("{:g} ", scale) : "") | color(theme.muted),
        }),
        separatorLight() | color(theme.track),
        hbox({
            text(compute_label) | color(theme.caption),
            text(std::format(" {:9.3f} MQ/s ", data.compute_m_dofs_per_sec)),
            filler(),
            sparkline(compute_history_, 14, theme.highlight),
        }),
        hbox({
            text("WALL") | color(theme.caption),
            text(std::format(" {:9.3f} MQ/s ", data.wall_m_dofs_per_sec)),
            filler(),
            sparkline(wall_history_, 14, theme.good),
        }),
        hbox({
            text("rate") | color(theme.caption),
            text(std::format(" {:9.2f} cycles/s", data.cycles_per_second)),
        }),
    });

    /*
     * Throughput history: an area chart of the most recent wall
     * throughput samples, one sample per half cell
     */

    auto history_graph = [this, scale](int width, int height) {
      /* A negative value leaves the column empty: */
      std::vector<int> values(width, -1);
      const auto n = std::min<std::size_t>(width, wall_history_.size());
      for (std::size_t i = 0; i < n; ++i) {
        const double value = wall_history_[wall_history_.size() - n + i];
        const auto level = static_cast<int>(std::round(value / scale * height));
        values[width - n + i] = std::clamp(level, 0, height);
      }
      return values;
    };

    auto history = hbox({
        vbox({
            text(std::format("{:g}", scale)) | color(theme.muted),
            filler(),
            text("0") | color(theme.muted),
        }) | size(WIDTH, EQUAL, 5),
        graph(std::move(history_graph)) | color(theme.good) | flex,
    });

    /*
     * Messages: the most recent status messages, scrolled to the newest
     */

    Elements message_lines;
    for (const auto &message : data.messages)
      message_lines.push_back(
          hbox({
              text(std::format("{:>6} ", message.cycle)) | color(theme.muted),
              text(message.text),
          }) |
          flex_shrink);
    if (message_lines.empty())
      message_lines.push_back(text("no messages") | color(theme.muted));

    auto messages = vbox(std::move(message_lines)) |
                    focusPositionRelative(0.f, 1.f) | yframe | at_most(5);

    /*
     * Time stepping
     */

    const auto counter = [](const std::string &name, const auto &n) {
      const auto c = [](unsigned int k) {
        return k == 0 ? text("0") | color(theme.muted)
                      : text(std::to_string(k)) | bold |
                            color(k < 10 ? theme.warn : theme.bad);
      };
      return hbox({
          text(name + " ") | color(theme.caption),
          c(n[0]),
          text("/") | color(theme.muted),
          c(n[1]),
          text("   "),
      });
    };

    Elements stepping = {
        hbox({
            label("scheme"),
            chip(data.time_stepping_scheme, theme.good),
            text("  CFL ") | color(theme.caption),
            text(std::format("{:.2f}", data.cfl)) | bold,
        }),
        hbox({
            label("tau"),
            text(std::format("{:.3e}", data.tau)) | bold,
            text(std::format("   ({:.3e} t/s)", data.time_per_second)) |
                color(theme.muted),
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
      stepping.push_back(hbox({
          label("GPU util"),
          bar(data.utilization, theme.bad, theme.good) | size(WIDTH, EQUAL, 24),
          text(std::format(" {:5.1f}%", 100. * data.utilization)) | bold,
          text(std::format("  on {} device{}",
                           data.n_devices,
                           data.n_devices == 1 ? "" : "s")) |
              color(theme.muted),
      }));
      const auto skew =
          std::clamp(safe(data.device_time_skew_fraction), 0., 1.);
      stepping.push_back(hbox({
          label("gpu skew"),
          gauge(static_cast<float>(skew)) | color(heat(4. * skew)) |
              bgcolor(theme.track) | size(WIDTH, EQUAL, 24),
          text(std::format(" {:5.1f}%", 100. * skew)) | bold,
          text(std::format("  ({:.2e} s/cycle) [p{}/{}]",
                           data.device_time_skew,
                           data.device_time_min_index,
                           data.device_time_max_index)) |
              color(theme.muted),
      }));
    } else {
      const auto skew = std::clamp(safe(data.cpu_time_skew_fraction), 0., 1.);
      stepping.push_back(hbox({
          label("cpu skew"),
          gauge(static_cast<float>(skew)) | color(heat(4. * skew)) |
              bgcolor(theme.track) | size(WIDTH, EQUAL, 24),
          text(std::format(" {:5.1f}%", 100. * skew)) | bold,
          text(std::format("  ({:.2e} s/cycle)", data.cpu_time_skew)) |
              color(theme.muted),
      }));
    }

    /*
     * Memory
     */

    const auto &memory = data.memory;
    auto memory_panel = vbox({
        hbox({
            text("min ") | color(theme.caption),
            text(std::format("{:9.1f}", memory.min)),
            text(std::format(" [p{}]", memory.min_index)) | color(theme.muted),
        }),
        hbox({
            text("avg ") | color(theme.caption),
            text(std::format("{:9.1f}", memory.avg)) | bold,
        }),
        hbox({
            text("max ") | color(theme.caption),
            text(std::format("{:9.1f}", memory.max)),
            text(std::format(" [p{}]", memory.max_index)) | color(theme.muted),
        }),
        range_bar(memory, nice_scale(memory.max), 32),
        hbox({
            text("0") | color(theme.muted),
            filler(),
            text(std::format("{:g} MiB", nice_scale(memory.max))) |
                color(theme.muted),
        }),
    });

    /*
     * Timers: a stacked colorbar of the time loop and a table
     */

    const int composition_width = screen_width - 4;
    Elements composition;
    std::vector<std::vector<Element>> table = {{
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
    unsigned int n_segment = 0;
    for (const auto &timer : data.timers) {
      const auto &wall = timer.wall_time;
      const double skew_negative =
          wall.avg > 0.
              ? std::max(100. * (wall.min - wall.avg) / wall.avg, -99.9)
              : 0.;
      const double skew_positive =
          wall.avg > 0.
              ? std::min(100. * (wall.max - wall.avg) / wall.avg, 99.9)
              : 0.;

      Element marker = text(" ");
      Element share = text("");
      Element percentage = text("");

      if (timer.fraction >= 0.) {
        const auto c = theme.segments[n_segment++ % theme.segments.size()];
        const auto fraction = std::clamp(safe(timer.fraction), 0., 1.);
        const auto length =
            static_cast<int>(std::round(fraction * composition_width));
        const int n = std::min(length, composition_width - used);
        if (n > 0) {
          composition.push_back(text(std::string(n, ' ')) | bgcolor(c));
          used += n;
        }
        marker = text("■ ") | color(c);
        share = gauge(static_cast<float>(fraction)) | color(c) |
                bgcolor(theme.track) | size(WIDTH, EQUAL, 20);
        percentage = text(std::format(" {:5.1f}%", 100. * fraction));
      }

      /* Every timer of the time loop is a "time step": */
      std::string name = timer.name;
      if (name.starts_with("time step "))
        name.erase(0, 10);

      table.push_back({
          marker,
          /* Truncate the name first if the terminal is too narrow: */
          text(name + "  ") | flex_shrink,
          text(std::format("{:.2f}s ", wall.avg)) | align_right,
          text(
              std::format(" {:5.1f}/{:4.1f}% ", skew_negative, skew_positive)) |
              color(heat(std::max(-skew_negative, skew_positive) / 25.)) |
              align_right,
          text(std::format(" p{}/p{} ", wall.min_index, wall.max_index)) |
              color(theme.muted),
          text(std::format("{:.2f}s ", timer.cpu_time)) | align_right,
          share,
          percentage,
      });
    }

    if (used < composition_width)
      composition.push_back(text(std::string(composition_width - used, ' ')) |
                            bgcolor(theme.track));

    auto timers = vbox({
        hbox(std::move(composition)),
        text(""),
        gridbox(std::move(table)),
    });

    /*
     * Tasks: the trigger of every task, when it fired last, and when it
     * fires next
     */

    std::vector<std::vector<Element>> task_table = {{
        text("task") | bold,
        text("  phase") | bold,
        text("  trigger") | bold,
        text("  last fired") | bold | flex_shrink,
        text("  ") | bold,
        text(" next") | bold,
    }};

    for (const auto &task : data.tasks) {
      const bool triggered = !task.trigger.empty();

      Element progress = text("");
      if (task.progress >= 0.)
        progress = gauge(static_cast<float>(task.progress)) |
                   color(theme.highlight) | bgcolor(theme.track) |
                   size(WIDTH, EQUAL, 16);

      task_table.push_back({
          text(task.name) | bold,
          hbox({text("  "), chip(task.phase, theme.highlight)}),
          text("  " + (triggered ? task.trigger : "--")) |
              color(triggered ? theme.caption : theme.muted),
          text("  " + (task.last.empty() ? "--" : task.last)) |
              color(task.last.empty() ? theme.muted : Color::Default) |
              flex_shrink,
          hbox({text("  "), progress}),
          text(" " + (task.next.empty() ? "--" : task.next)) |
              color(task.next.empty() ? theme.muted : theme.good),
      });
    }

    /*
     * Footer
     */

    auto footer = hbox({
        text(" terminal update every ") | color(theme.muted),
        text(std::format("{:.2f}s", data.terminal_update_interval)) |
            color(theme.caption),
        filler(),
        text(version_string() + " ") | color(theme.muted),
    });

    /*
     * Assemble the dashboard:
     */

    auto document = vbox({
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
        panel("timers", timers),
        panel("tasks", gridbox(std::move(task_table))),
        footer,
    });

    /*
     * Render the dashboard with its natural height, or stretched to fill
     * the terminal up to the maximal height:
     */

    auto dashboard = document | size(WIDTH, EQUAL, screen_width);
    int height = Dimension::Fit(dashboard, true).dimy;
    if (settings.height > 0)
      height = std::max(
          height,
          std::min(static_cast<int>(settings.height), terminal_height_));

    auto image = Screen::Create(Dimension::Fixed(screen_width),
                                Dimension::Fixed(height));
    Render(image, dashboard | size(HEIGHT, EQUAL, height));

    /*
     * Place the dashboard on a screen that covers the whole terminal (or
     * more if the dashboard does not fit):
     */

    const int rows = std::max(height, terminal_height_);
    const int free_columns = terminal_columns - screen_width;
    const int free_rows = rows - height;

    const auto [left, top] =
        position_.offset(settings.location, free_columns, free_rows);

    stage_ = std::make_unique<animation::Stage>(
        Screen(terminal_columns, rows),
        Box{left, left + screen_width - 1, top, top + height - 1},
        theme.animation);
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < screen_width; ++x)
        stage_->screen.PixelAt(left + x, top + y) = image.PixelAt(x, y);

    /* Leave the screen around the final dashboard empty: */
    if (animation_ && !data.final_time)
      animation_->draw(*stage_, data);

    /*
     * Redraw the whole terminal in place, after clearing the screen if the
     * terminal has been resized or for the final snapshot:
     */
    const bool clear = resized || data.final_time;
    std::string output = clear ? "\033[2J\033[H" : "\033[H";
    output += stage_->screen.ToString();

    /* If the dashboard does not fit, clear whatever is left below: */
    if (rows > terminal_height_)
      output += "\n\033[J";

    return output;
  }


  std::string Dashboard::render_animation()
  {
    theme = make_palette(settings_.color_scheme);

    const auto [columns, lines] = terminal_size();
    terminal_width_ = columns;
    terminal_height_ = lines;
    /* A stage with an empty dashboard: */
    stage_ = std::make_unique<animation::Stage>(
        Screen(columns, lines), Box{0, -1, 0, -1}, theme.animation);

    if (animation_)
      animation_->draw(*stage_, data_);

    return "\033[2J\033[H" + stage_->screen.ToString();
  }


  std::string Dashboard::step()
  {
    /* Nothing to do without animation, or if the last frame has scrolled: */
    if (!animation_ || stage_->screen.dimy() > terminal_height_)
      return "";

    stage_->changed.clear();
    animation_->step(*stage_, data_);
    return changes(*stage_);
  }


  void Dashboard::animate()
  {
    constexpr auto interval = std::chrono::milliseconds(150);

    std::unique_lock lock(mutex_);
    while (!condition_.wait_for(lock, interval, [this] { return stop_; })) {
      /*
       * Redraw the last frame right away if the terminal is resized, or
       * if the dashboard moves:
       */
      const auto [columns, lines] = terminal_size();
      const bool resized =
          columns != terminal_width_ || lines != terminal_height_;
      if (animation_only_ && resized)
        write(render_animation());
      else if (!animation_only_ &&
               (resized || position_.move(settings_.location)))
        write(render(data_, settings_));
      else
        write(step());
    }
  }


  void Dashboard::stop()
  {
    {
      std::scoped_lock lock(mutex_);
      stop_ = true;
    }
    condition_.notify_all();
    if (thread_.joinable())
      thread_.join();
  }


  Dashboard::Dashboard() = default;


  Dashboard::~Dashboard()
  {
    stop();
    if (terminal_ >= 0)
      ::close(terminal_);
  }


  std::pair<int, int> Dashboard::terminal_size() const
  {
    winsize size{};
    if (terminal_ >= 0 && ::ioctl(terminal_, TIOCGWINSZ, &size) == 0 &&
        size.ws_col > 0 && size.ws_row > 0)
      return {size.ws_col, size.ws_row};

    const auto fallback = Terminal::Size();
    return {fallback.dimx, fallback.dimy};
  }


  void Dashboard::write(const std::string &output) const
  {
    if (terminal_ < 0) {
      std::cout << output << std::flush;
      return;
    }

    const char *data = output.data();
    std::size_t remaining = output.size();
    while (remaining > 0) {
      const auto n = ::write(terminal_, data, remaining);
      if (n < 0) {
        if (errno == EINTR)
          continue;
        return;
      }
      data += n;
      remaining -= static_cast<std::size_t>(n);
    }
  }


  void Dashboard::start(const DashboardSettings &settings)
  {
    std::scoped_lock lock(mutex_);

    terminal_ = ::open("/dev/tty", O_WRONLY | O_NOCTTY | O_CLOEXEC);
    settings_ = settings;
    animation_ = animation::create(settings.animation);
    animation_only_ = true;

    /* Write out everything that is still pending on stdout first: */
    std::cout << std::flush;
    write(render_animation());

    thread_ = std::thread([this] { animate(); });
  }


  void Dashboard::show(const DashboardData &data,
                       const DashboardSettings &settings)
  {
    /* No more updates after the final snapshot: */
    if (data.final_time)
      stop();

    std::scoped_lock lock(mutex_);

    /* Update the throughput history: */

    constexpr std::size_t history_size = 512;
    wall_history_.push_back(safe(data.wall_m_dofs_per_sec));
    compute_history_.push_back(safe(data.compute_m_dofs_per_sec));
    if (wall_history_.size() > history_size) {
      wall_history_.pop_front();
      compute_history_.pop_front();
    }

    data_ = data;
    settings_ = settings;
    animation_only_ = false;
    position_.move(settings_.location);

    /* Write out everything that is still pending on stdout first: */
    std::cout << std::flush;

    /* Continue with subsequent output below the final dashboard: */
    write(render(data_, settings_) + (data.final_time ? "\n" : ""));
  }
} // namespace ryujin
