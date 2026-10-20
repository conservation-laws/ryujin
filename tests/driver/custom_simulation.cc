//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/driver/scheduler.h>
#include <ryujin/interface/task.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

//
// Test the Scheduler with a completely different primary task: the
// ToySimulation solves the ordinary differential equation y' = -y with
// the explicit Euler method and does not use any class of the
// discretization. A task "print" prints the solution whenever its trigger
// fires (by default at every timer tick).
//

using namespace ryujin;
using namespace dealii;

using Number = NUMBER;


class ToySimulation final : public Interface::Task<Number>
{
public:
  ToySimulation()
      : Interface::Task<Number>("toy simulation", TaskPhase::step)
  {
  }

  void prepare(TaskContext<Number> & /*context*/) override
  {
    y_ = Number(1.);
    n_steps_ = 0;
  }

  void execute(TaskContext<Number> &context) override
  {
    context.tau = std::min(Number(0.01), context.t_max - context.t);
    y_ -= context.tau * y_;
    ++n_steps_;
  }

  Number y_ = Number(0.);
  unsigned int n_steps_ = 0;
};


class PrintTask final : public ParameterAcceptor, public Interface::Task<Number>
{
public:
  PrintTask(const ToySimulation &simulation)
      : ParameterAcceptor("/B - PrintTask")
      , Interface::Task<Number>("print", TaskPhase::post_step)
      , simulation_(simulation)
  {
    trigger_.add_parameters(*this);
  }

  void execute(TaskContext<Number> &context) override
  {
    if (!trigger_.fire(context))
      return;

    std::cout << "print: t = " << std::fixed << std::setprecision(6)
              << context.t << ", y = " << simulation_.y_ << ", steps "
              << simulation_.n_steps_ << std::endl;
  }

  void finalize(TaskContext<Number> &context) override
  {
    std::cout << "finalize: cycles " << context.cycle << ", t = " << std::fixed
              << std::setprecision(6) << context.t << std::endl;
  }

private:
  Trigger<Number> trigger_;
  const ToySimulation &simulation_;
};


int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  ToySimulation simulation;
  PrintTask print(simulation);
  Scheduler<Number> scheduler(
      MPI_COMM_WORLD, {&simulation}, {&print}, [](const std::string &) {});

  std::stringstream parameters;
  parameters << "subsection A - Scheduler\n"
             << "  set final time        = 1.0\n"
             << "  set timer granularity = 0.25\n"
             << "  set tasks             = print\n"
             << "end\n";
  ParameterAcceptor::initialize(parameters);

  scheduler.run();

  return 0;
}
