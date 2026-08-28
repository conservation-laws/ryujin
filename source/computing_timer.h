//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include "instrumentation.h"
#include <compile_time_options.h>
#include <convenience_macros.h>
#include <simd.h>

#include <deal.II/base/timer.h>

#include <map>
#include <string>

#ifdef DEBUG_OUTPUT
#include <iostream>
#endif

namespace ryujin
{
  /**
   * An accumulator for the wall and CPU time spent in named sections of
   * the host code, keyed by section name.
   *
   * @note This class is a singleton and not thread safe.
   *
   * @ingroup Miscellaneous
   */
  class ComputingTimer
  {
  public:
    /**
     * A RAII scope for the timer of the specified @p section.
     *
     * The constructor of the class starts a timer with the specified name.
     * If ryujin is configured with likwid then likwid instrumentation will
     * also be started with the given section name. The destructor of the
     * class stops the timer again and also stops likwid instrumentation.
     *
     * @note This class does not perform MPI synchronization in contrast to
     * the deal.II counterpart.
     */
    class Scope
    {
    public:
      /**
       * Constructor. Starts a timer for the selected @p section.
       */
      Scope(const std::string &section)
          : section_(section)
      {
        LIKWID_MARKER_START(section_.c_str());
        timers_[section_].start();
#ifdef DEBUG_OUTPUT
        std::cout << "{scoped timer} \"" << section_ << "\" started"
                  << std::endl;
#endif
      }

      /**
       * Destructor. Stops the timer.
       */
      ~Scope()
      {
#ifdef DEBUG_OUTPUT
        std::cout << "{scoped timer} \"" << section_ << "\" stopped"
                  << std::endl;
#endif
        timers_[section_].stop();
        LIKWID_MARKER_STOP(section_.c_str());
      }

    private:
      const std::string section_;
    };

    /**
     * Return a reference to the deal.II timer object for the given
     * @p section, creating it if necessary.
     */
    static dealii::Timer &timer(const std::string &section)
    {
      return timers_[section];
    }

    /**
     * Return a const reference to the full map of named timers.
     */
    static const std::map<std::string, dealii::Timer> &timers()
    {
      return timers_;
    }

    /**
     * Reset the timer object back to zero
     */
    static void reinit()
    {
      timers_.clear();
    }

  private:
    static inline std::map<std::string, dealii::Timer> timers_;
  };


  /*
   * An accumulator for the wall time spent executing compute kernels on the
   * device.
   *
   * All device compute loops (defined in loop.h) launch a kernel and then
   * fence the execution space, see gpu_loop(). The wall time spent in the
   * launch and in the subsequent fence is thus a good proxy for the time
   * the device is actually busy. (Some quick comparison with a profiler
   * suggest a discrepancy of less than 5%.)
   *
   * @note We only measure compute loops with this class. Kokkos-internal
   * operations, such as the deep copies, as well as the exchange buffer
   * kernel of the SparseMatrix class, do not enter this calcuation.
   * (Together they contribute about another 4% of device runtime.)
   *
   * @note This class is a singleton and not thread safe.
   *
   * @ingroup Miscellaneous
   */
  class DeviceTimer
  {
  public:
    /**
     * A Scope class that adds the wall time of its own lifetime to the
     * accumulated device time and increments the kernel counter.
     */
    class Scope
    {
    public:
      Scope()
      {
        timer_.start();
      }

      ~Scope()
      {
        timer_.stop();
        n_kernels_ += 1;
      }
    };

    /**
     * The accumulated wall time (in seconds) spent executing device
     * kernels. The value increases monotonically since the last reinit().
     */
    static double seconds()
    {
      return timer_.wall_time();
    }

    /**
     * The number of device kernels that have been launched so far.
     */
    static std::uint64_t n_kernels()
    {
      return n_kernels_;
    }

    /**
     * Reset the accumulated device time and the kernel counter back to zero.
     */
    static void reinit()
    {
      timer_.reset();
      n_kernels_ = 0;
    }

  private:
    static inline dealii::Timer timer_;
    static inline std::uint64_t n_kernels_ = 0;
  };
} // namespace ryujin
