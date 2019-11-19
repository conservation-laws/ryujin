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
    }

    ~Scope()
    {
      computing_timer_[section_].stop();
    }

  private:
    std::map<std::string, dealii::Timer> &computing_timer_;
    const std::string section_;
  };
} // namespace grendel

#endif /* SCOPE_H */
