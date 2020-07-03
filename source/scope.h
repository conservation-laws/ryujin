//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef SCOPE_H
#define SCOPE_H

#include <deal.II/base/timer.h>

#include <map>
#include <string>

namespace ryujin
{
  /**
   * A RAII scope for deal.II timer objects.
   *
   * This class does not perform MPI synchronization in contrast to the
   * deal.II counterpart.
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
    }

  private:
    std::map<std::string, dealii::Timer> &computing_timer_;
    const std::string section_;
  };
} // namespace ryujin

#endif /* SCOPE_H */
