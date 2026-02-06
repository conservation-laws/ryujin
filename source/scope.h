//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2025 by the ryujin authors
//

#pragma once

#include "instrumentation.h"
#include <compile_time_options.h>

#include <deal.II/base/timer.h>

#include <map>
#include <string>

#ifdef DEBUG_OUTPUT
#include <iostream>
#endif

namespace ryujin
{
  /**
   * A RAII scope for deal.II timer objects and likwid instrumentation.
   *
   * The constructor of the class starts a timer with the specified name.
   * If ryujin is configured with likwid then likwid instrumentation will
   * also be started with the given section name. The destructor of the
   * class stops the timer again and also stops likwid instrumentation.
   *
   * @note This class does not perform MPI synchronization in contrast to
   * the deal.II counterpart.
   *
   * @ingroup Miscellaneous
   */
  class Scope
  {
  public:
    /**
     * Constructor. Starts a timer for the selected @p section.
     */
    Scope(std::map<std::string, dealii::Timer> &computing_timer,
          const std::string &section)
        : computing_timer_(computing_timer)
        , section_(section)
    {
      LIKWID_MARKER_START(section_.c_str());
      computing_timer_[section_].start();
#ifdef DEBUG_OUTPUT
      std::cout << "{scoped timer} \"" << section_ << "\" started" << std::endl;
#endif
    }

    /**
     * Destructor. Stops the timer.
     */
    ~Scope()
    {
#ifdef DEBUG_OUTPUT
      std::cout << "{scoped timer} \"" << section_ << "\" stopped" << std::endl;
#endif
      computing_timer_[section_].stop();
      LIKWID_MARKER_STOP(section_.c_str());
    }

  private:
    std::map<std::string, dealii::Timer> &computing_timer_;
    const std::string section_;
  };
} // namespace ryujin
