//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#include <driver/dashboard_text.h>

#include <deal.II/base/utilities.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ryujin
{
  namespace
  {
    void print_head(const DashboardData &data, std::ostream &stream)
    {
      std::ostringstream primary;
      if (data.final_time) {
        primary << "FINAL  (cycle "
                << dealii::Utilities::int_to_string(data.cycle, 6) << ")";
      } else {
        primary << "Cycle  " << dealii::Utilities::int_to_string(data.cycle, 6)
                << "  (" << std::fixed << std::setprecision(1)
                << data.t / data.t_final * 100 << "%)";
      }

      std::ostringstream secondary;
      secondary << "at time t = " << std::setprecision(8) << std::fixed
                << data.t;

      const auto pad = [](const std::string &text) {
        const int size = text.size();
        return std::string(std::max(0, 34 - size) / 2, ' ') + text +
               std::string(std::max(0, 35 - size) / 2, ' ');
      };

      /* clang-format off */
      stream << "\n";
      stream << "    ####################################################\n";
      stream << "    #########" << pad(primary.str())   << "#########\n";
      stream << "    #########" << pad(secondary.str()) << "#########\n";
      stream << "    ####################################################\n";
      stream << std::endl;
      /* clang-format on */
    }


    void print_information(const DashboardData &data, std::ostream &stream)
    {
      stream << "Information: (HYP) " << data.hyperbolic_name;
      if (!data.parabolic_name.empty())
        stream << "\n             (PAR) " << data.parabolic_name;
      stream << "\n             [" << data.base_name << "] ";
      if (data.n_ensembles > 1)
        stream << data.n_ensembles << " ensembles ";
      stream << "with " << static_cast<std::uint64_t>(data.n_global_dofs)
             << " Qdofs on " << data.n_ranks << " ranks "
             << (data.threads.empty() ? "" : "/ " + data.threads + " ") << "<"
             << data.backend << ">\n";

      stream << "             [ tasks: ";
      for (const auto &task : data.tasks)
        stream << task.name << (&task == &data.tasks.back() ? " " : ", ");
      stream << "]\n";
    }


    void print_memory_statistics(const DashboardData &data,
                                 std::ostream &stream)
    {
      const auto &memory = data.memory;
      const unsigned int n = dealii::Utilities::needed_digits(data.n_ranks);

      std::ostringstream output;
      output << "\nMemory:      [MiB]"                            //
             << std::setw(8) << memory.min                        //
             << " [p" << std::setw(n) << memory.min_index << "] " //
             << std::setw(8) << memory.avg << " "                 //
             << std::setw(8) << memory.max                        //
             << " [p" << std::setw(n) << memory.max_index << "]"; //

      stream << output.str() << std::endl;
    }


    void print_timers(const DashboardData &data, std::ostream &stream)
    {
      std::vector<std::ostringstream> output(data.timers.size());

      /* Pad all lines to the same length: */
      const auto equalize = [&]() {
        std::size_t length = 0;
        for (const auto &it : output)
          length = std::max(length, it.str().length());
        for (auto &it : output)
          it << std::string(length - it.str().length() + 1, ' ');
      };

      for (std::size_t i = 0; i < output.size(); ++i)
        output[i] << "  " << data.timers[i].name;
      equalize();

      const unsigned int n = dealii::Utilities::needed_digits(data.n_ranks);
      for (std::size_t i = 0; i < output.size(); ++i) {
        const auto &wall_time = data.timers[i].wall_time;

        constexpr auto eps = std::numeric_limits<double>::epsilon();
        /*
         * Cut off at 99.9% to avoid silly percentages cluttering up the
         * output.
         */
        const auto skew_negative = std::max(
            100. * (wall_time.min - wall_time.avg) / wall_time.avg - eps,
            -99.9);
        const auto skew_positive = std::min(
            100. * (wall_time.max - wall_time.avg) / wall_time.avg + eps, 99.9);

        output[i] << std::setprecision(2) << std::fixed << std::setw(9)
                  << wall_time.avg << "s [sk: " << std::setprecision(1)
                  << std::setw(5) << std::fixed << skew_negative << "%/"
                  << std::setw(4) << std::fixed << skew_positive << "%]";
        output[i] << " [p" << std::setw(n) << wall_time.min_index << "/"
                  << wall_time.max_index << "]";
      }
      equalize();

      for (std::size_t i = 0; i < output.size(); ++i) {
        const auto &timer = data.timers[i];
        output[i] << std::setprecision(2) << std::fixed << std::setw(12)
                  << timer.cpu_time << "s ";
        if (timer.fraction >= 0.)
          output[i] << "(" << std::setprecision(1) << std::setw(4)
                    << 100. * timer.fraction << "%)";
      }
      equalize();

      stream << std::endl << "Timer statistics:\n";
      for (const auto &it : output)
        stream << it.str() << std::endl;
    }


    void print_tasks(const DashboardData &data, std::ostream &stream)
    {
      std::size_t name_width = 0;
      std::size_t trigger_width = 0;
      std::size_t last_width = 0;
      for (const auto &task : data.tasks) {
        name_width = std::max(name_width, task.name.length());
        trigger_width = std::max(trigger_width, task.trigger.length());
        last_width = std::max(last_width, task.last.length());
      }

      std::ostringstream output;
      output << "\nTasks:\n";
      for (const auto &task : data.tasks) {
        output << "  " << std::left << std::setw(name_width) << task.name
               << "  [" << std::setw(9) << task.phase << "]";
        if (!task.trigger.empty())
          output << "  " << std::setw(trigger_width) << task.trigger
                 << "  last: " << std::setw(last_width)
                 << (task.last.empty() ? "--" : task.last)
                 << "  next: " << (task.next.empty() ? "--" : task.next);
        output << std::right << "\n";
      }

      stream << output.str() << std::flush;
    }


    void print_throughput(const DashboardData &data, std::ostream &stream)
    {
      /* Print Jean-Luc and Martin metrics: */

      std::ostringstream output;

      /* clang-format off */
      output << std::endl;

      output << "Throughput:" << std::endl;

      if (data.device) {
        /* Write out some statistics about GPU utilization: */

        output << "  GPU : "
               << std::setprecision(4) << std::fixed << data.compute_m_dofs_per_sec
               << " MQ/s in compute kernels (on "
               << data.n_devices << (data.n_devices == 1 ? " device)" : " devices)")
               << std::endl;

        output << "        ["
               << std::setprecision(2) << std::fixed << data.delta_device_time
               << "s device / "
               << std::setprecision(2) << std::fixed << data.delta_wall_time
               << "s wall = "
               << std::setprecision(1) << std::fixed << 100. * data.utilization
               << "% utilization ]" << std::endl;

        output << "        [device time skew: "
               << std::setprecision(2) << std::scientific << data.device_time_skew
               << "s/cycle ("
               << std::setprecision(1) << std::setw(4) << std::setfill(' ') << std::fixed
               << 100. * data.device_time_skew_fraction
               << "%) [p" << data.device_time_min_index
               << "/" << data.device_time_max_index << "]]" << std::endl;

      } else {
        /* Write out some statistics about CPU utilization: */

        output << "  "
               << (data.fudged_cpu_timings ? "CPU*: " : "CPU : ")
               << std::setprecision(4) << std::fixed << data.compute_m_dofs_per_sec
               << " MQ/s  ("
               << std::scientific << 1. / data.compute_m_dofs_per_sec * 1.e-6
               << " s/Qdof/substep)" << std::endl;

        output << "        [cpu time skew: "
               << std::setprecision(2) << std::scientific << data.cpu_time_skew
               << "s/cycle ("
               << std::setprecision(1) << std::setw(4) << std::setfill(' ') << std::fixed
               << 100. * data.cpu_time_skew_fraction
               << "%)]" << std::endl;
      }

      output << "  WALL: "
             << std::setprecision(4) << std::fixed << data.wall_m_dofs_per_sec
             << " MQ/s  ("
             << std::scientific << 1. / data.wall_m_dofs_per_sec * 1.e-6
             << " s/Qdof/substep)  ("
             << std::setprecision(2) << std::fixed << data.cycles_per_second
             << " cycles/s)" << std::endl;

      output << "        [ " << data.time_stepping_scheme
             << " with CFL = " << std::setprecision(2) << std::fixed << data.cfl
             << " (" << data.restarts[0] << "/" << data.restarts[1] << " rsts)"
             << " (" << data.warnings[0] << "/" << data.warnings[1] << " warn)"
             << " (" << data.corrections[0] << "/" << data.corrections[1] << " corr) ]"
             << std::endl;

      for (const auto &line : data.solver_statistics)
        output << "        [ " << line << " ]" << std::endl;

      output << "        [ dt = "
             << std::scientific << std::setprecision(2) << data.tau
             << " ( "
             << data.time_per_second
             << " dt/s) ]" << std::endl;
      /* clang-format on */

      /* And print an ETA: */

      auto eta = static_cast<unsigned int>(data.eta);

      output << "\n  ETA : ";

      const unsigned int days = eta / (24 * 3600);
      if (days > 0) {
        output << days << " d  ";
        eta %= 24 * 3600;
      }

      const unsigned int hours = eta / 3600;
      if (hours > 0) {
        output << hours << " h  ";
        eta %= 3600;
      }

      const unsigned int minutes = eta / 60;
      output << minutes << " min";

      output << "   (terminal update every " //
             << std::setprecision(2) << std::fixed
             << data.terminal_update_interval << "s)";

      stream << output.str() << std::endl;
    }
  } // namespace


  std::string render_text(const DashboardData &data)
  {
    std::ostringstream output;
    print_head(data, output);
    print_information(data, output);
    print_memory_statistics(data, output);
    print_timers(data, output);
    print_tasks(data, output);
    print_throughput(data, output);
    return output.str();
  }
} // namespace ryujin
