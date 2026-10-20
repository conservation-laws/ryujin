//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/driver/simulation.h>
#include <ryujin/driver/task_composition.h>
#include <ryujin/euler/hyperbolic_description.h>
#include <ryujin/euler_poisson/parabolic_description.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

//
// Test checkpointing of the dynamic state: We run the isentropic vortex
// with the Gauss law restart strategy "no restart" once uninterrupted to
// the final time, and once split into a first run to half of the final
// time that writes a checkpoint, and a second run that resumes from it.
// The potential is only carried by the state vector, so the resumed
// computation agrees with the uninterrupted one only if the potential is
// written to and read from the checkpoint.
//

using namespace ryujin;
using namespace dealii;

using HyperbolicDescription = Euler::HyperbolicDescription;
using ParabolicDescription = EulerPoisson::ParabolicDescription;

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
  parameters << "subsection A - Scheduler\n"
             << time_loop_parameters             //
             << "  set tasks = error evaluation" //
             << (checkpointing_parameters.empty() ? "" : ", checkpointing")
             << "\n"                                      //
             << "end\n"                                   //
             << "subsection M - Logger\n"                 //
             << "  set terminal update interval = 0\n"    //
             << "end\n"                                   //
             << "subsection N - Checkpointing\n"          //
             << checkpointing_parameters                  //
             << "end\n"                                   //
             << "subsection B - Equation\n"               //
             << "  set alpha                       = 1\n" //
             << "  set electrostatic configuration = function\n"
             << "  subsection function\n"
             << "    set time dependent   = true\n"
             << "    set magnetic field z = 0.\n"
             << "    set background density = -pow(1.-(1.4-1.)/(2 * 1.4) * "
                "pow(5. / (2. * 3.1415926535897932) * exp(0.5 * (1. - pow(x "
                "+ 1. - 0.707106781188 * t, 2.) - pow(y + 1. - 0.707106781188 "
                "* t, 2.))), 2.), 1. / (1.4 - 1.))\n"
             << "  end\n"
             << "end\n"
             << "subsection C - Discretization\n"
             << "  set finite element ansatz = cG Q1\n"
             << "  set geometry        = rectangular domain\n"
             << "  set mesh refinement = 4\n"
             << "  subsection rectangular domain\n"
             << "    set boundary condition bottom = dirichlet\n"
             << "    set boundary condition left   = dirichlet\n"
             << "    set boundary condition right  = dirichlet\n"
             << "    set boundary condition top    = dirichlet\n"
             << "    set position bottom left      = -5, -5\n"
             << "    set position top right        =  5,  5\n"
             << "  end\n"
             << "end\n"
             << "subsection E - InitialValues\n"
             << "  set configuration = isentropic vortex\n"
             << "  set direction     =  1,  1\n"
             << "  set position      = -1, -1\n"
             << "  subsection isentropic vortex\n"
             << "    set mach number = 1\n"
             << "    set beta        = 5\n"
             << "  end\n"
             << "end\n"
             << "subsection G - ParabolicModule\n"
             << "  set gauss law restart strategy = no restart\n"
             << "  set tolerance                  = 1e-12\n"
             << "end\n"
             << "subsection H - TimeIntegrator\n"
             << "  set cfl min               = 0.2\n"
             << "  set cfl max               = 0.2\n"
             << "  set cfl recovery strategy = none\n"
             << "  set time stepping scheme  = strang erk 33 cn\n"
             << "end\n";

  {
    Simulation<HyperbolicDescription, ParabolicDescription, dim, Number>
        simulation(MPI_COMM_WORLD);
    TaskComposition<dim, Number> task_composition(simulation);
    ParameterAcceptor::initialize(parameters);
    task_composition.run();
  }

  ParameterAcceptor::clear();
}


/*
 * Return the last line of the error log of a computation.
 */
std::string last_error_line(const std::string &base_name)
{
  std::ifstream file(base_name + "-error.log");
  std::string line, last_line;
  while (std::getline(file, line))
    if (!line.empty() && line[0] != '#')
      last_line = line;
  return last_line;
}


int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  /* An uninterrupted computation: */
  run("  set basename         = full\n"
      "  set final time       = 0.5\n"
      "  set timer granularity = 0.5\n");

  /* A computation to half of the final time that writes a checkpoint: */
  run("  set basename         = split\n"
      "  set final time       = 0.25\n"
      "  set timer granularity = 0.25\n",
      "  set trigger interval = 1.e-6\n");

  /* Resume the computation and run to the final time: */
  run("  set basename         = split\n"
      "  set final time       = 0.5\n"
      "  set timer granularity = 0.25\n",
      "  set resume = true\n");

  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    const auto full = last_error_line("full");
    const auto split = last_error_line("split");
    std::cout << "\nfinal errors (time, Linf, L1, L2):\n"
              << "  uninterrupted: " << full << "\n"
              << "  resumed:       " << split << std::endl;
  }

  return 0;
}
