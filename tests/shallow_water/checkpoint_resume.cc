//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/driver/simulation.h>
#include <ryujin/driver/task_composition.h>
#include <ryujin/shallow_water/hyperbolic_description.h>
#include <ryujin/stub/parabolic_module.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

//
// Test that resuming from a checkpoint reproduces an uninterrupted
// computation on a locally refined mesh: A bathymetry with alternating
// dry and wet regions crosses the refinement interface. We run once
// uninterrupted to the final time, and once split into a first run that
// writes a checkpoint and a second run that resumes from it.
//

using namespace ryujin;
using namespace dealii;

using HyperbolicDescription = ShallowWater::HyperbolicDescription;
using ParabolicDescription = StubParabolicDescription;

constexpr int dim = 2;
using Number = NUMBER;


/*
 * Run a TaskComposition with the given time loop parameters, with
 * checkpointing if @p checkpointing_parameters is not empty.
 */
void run(const std::string &time_loop_parameters,
         const std::string &checkpointing_parameters = "")
{
  std::stringstream parameters;
  parameters
      << "subsection A - Scheduler\n"
      << time_loop_parameters                                                //
      << "  set tasks = error evaluation"                                    //
      << (checkpointing_parameters.empty() ? "" : ", checkpointing") << "\n" //
      << "end\n"                                                             //
      << "subsection M - Logger\n"                                           //
      << "  set terminal update interval = 0\n"                              //
      << "end\n"                                                             //
      << "subsection N - Checkpointing\n"                                    //
      << checkpointing_parameters                                            //
      << "end\n"                                                             //
      << "subsection B - Equation\n"
      << "  set gravity               = 9.81\n"
      << "  set reference water depth = 1\n"
      << "end\n"
      << "subsection C - Discretization\n"
      << "  set finite element ansatz      = cG Q1\n"
      << "  set geometry                   = rectangular domain\n"
      << "  set mesh refinement            = 3\n"
      << "  set mesh local refinement      = 1\n"
      << "  set mesh local refinement region = y\n"
      << "  subsection rectangular domain\n"
      << "    set boundary condition bottom = slip\n"
      << "    set boundary condition left   = slip\n"
      << "    set boundary condition right  = slip\n"
      << "    set boundary condition top    = slip\n"
      << "    set position bottom left      = -1, -1\n"
      << "    set position top right        =  1,  1\n"
      << "  end\n"
      << "end\n"
      << "subsection E - InitialValues\n"
      << "  set configuration = function\n"
      << "  subsection function\n"
      << "    set bathymetry expression      = if(sin(11 * x) > 0, 0.6, 0)\n"
      << "    set water elevation expression = 0.5\n"
      << "    set velocity x expression      = 0.1\n"
      << "    set velocity y expression      = 0.05\n"
      << "  end\n"
      << "end\n"
      << "subsection H - TimeIntegrator\n"
      << "  set cfl min               = 0.5\n"
      << "  set cfl max               = 0.5\n"
      << "  set cfl recovery strategy = none\n"
      << "  set time stepping scheme  = erk 33\n"
      << "end\n";

  {
    Simulation<dim, dim, Number, HyperbolicDescription, ParabolicDescription>
        simulation(MPI_COMM_WORLD);
    TaskComposition<dim, dim, Number> task_composition(simulation);
    ParameterAcceptor::initialize(parameters);
    task_composition.run();
  }

  ParameterAcceptor::clear();
}


/*
 * Return the last line of the error log of a computation as numbers.
 */
std::vector<double> last_error_line(const std::string &base_name)
{
  std::ifstream file(base_name + "-error.log");
  std::string line, last_line;
  while (std::getline(file, line))
    if (!line.empty() && line[0] != '#')
      last_line = line;

  std::vector<double> result;
  std::istringstream stream(last_line);
  for (double value; stream >> value;)
    result.push_back(value);
  return result;
}


int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  /* An uninterrupted computation: */
  run("  set basename          = full\n"
      "  set final time        = 0.4\n"
      "  set timer granularity = 0.4\n");

  /* A computation to half of the final time that writes a checkpoint: */
  run("  set basename          = split\n"
      "  set final time        = 0.2\n"
      "  set timer granularity = 0.2\n",
      "  set trigger interval = 1.e-6\n");

  /* Resume the computation and run to the final time: */
  run("  set basename          = split\n"
      "  set final time        = 0.4\n"
      "  set timer granularity = 0.2\n",
      "  set resume = true\n");

  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    const auto full = last_error_line("full");
    const auto split = last_error_line("split");

    bool agree = full.size() >= 4 && full.size() == split.size();
    for (std::size_t i = 0; agree && i < full.size(); ++i)
      agree = std::isfinite(full[i]) &&
              std::abs(full[i] - split[i]) <= 1.e-10 * std::abs(full[i]);

    std::cout << "\nresumed computation agrees with uninterrupted one: "
              << (agree ? "true" : "false") << std::endl;
  }

  return 0;
}
