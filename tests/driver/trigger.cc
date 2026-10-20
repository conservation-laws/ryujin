//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/interface/task.h>

#include <iostream>
#include <sstream>

//
// Test the Trigger: every n-th cycle, every n-th timer tick, wall clock
// intervals, and time points (configured with run time parameters).
//

using namespace ryujin;
using Number = double;

class TriggerParameters : public dealii::ParameterAcceptor
{
public:
  TriggerParameters()
      : ParameterAcceptor("/Test")
  {
    trigger.add_parameters(*this);
  }

  Trigger<Number> trigger;
};


int main()
{
  TaskContext<Number> context;
  context.timer_granularity = 0.1;

  {
    std::cout << "every cycle, multiplier 3:";
    Trigger<Number> trigger(TriggerMode::every_cycle, 3);
    for (context.cycle = 1; context.cycle <= 10; ++context.cycle)
      if (trigger.fire(context))
        std::cout << " " << context.cycle;
    std::cout << std::endl;
  }

  {
    std::cout << "timer tick, multiplier 2:";
    Trigger<Number> trigger(TriggerMode::timer_tick, 2);
    context.timer_cycle = 0;
    for (context.t = 0.; context.t < 1.; context.t += 0.04) {
      context.timer_tick = context.tick_reached();
      if (trigger.fire(context))
        std::cout << " " << context.timer_cycle;
      if (context.timer_tick)
        ++context.timer_cycle;
    }
    std::cout << std::endl;
  }

  {
    std::cout << "wall clock, interval 1.5:";
    Trigger<Number> trigger(TriggerMode::wall_clock, 1, 1.5);
    for (context.wall_time = 0.; context.wall_time < 6.;
         context.wall_time += 0.5)
      if (trigger.fire(context))
        std::cout << " " << context.wall_time;
    std::cout << std::endl;
  }

  {
    std::cout << "time points:";
    TriggerParameters parameters;
    std::istringstream input("subsection Test\n"
                             "  set trigger             = time points\n"
                             "  set trigger time points = 0.7, 0.1, 0.3, 0.35\n"
                             "end\n");
    dealii::ParameterAcceptor::initialize(input);
    for (context.t = 0.25; context.t < 1.; context.t += 0.1)
      if (parameters.trigger.fire(context))
        std::cout << " " << context.t;
    std::cout << std::endl;
  }
}
