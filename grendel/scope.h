//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef SCOPE_H
#define SCOPE_H

#include <deal.II/base/timer.h>

#include <map>
#include <string>

namespace grendel
{
  class Scope
  {
  public:
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
} // namespace grendel

#endif /* SCOPE_H */
