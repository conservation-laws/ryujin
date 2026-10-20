//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/discretization/mesh_adaptor.h>
#include <ryujin/driver/checkpointing.h>
#include <ryujin/driver/logger.h>
#include <ryujin/driver/scheduler.h>
#include <ryujin/interface/simulation.h>
#include <ryujin/output/error_evaluation.h>
#include <ryujin/output/quantities.h>
#include <ryujin/output/vtu_output.h>

namespace ryujin
{
  /**
   * A TaskComposition assembles the library tasks and the Scheduler around
   * a Simulation. It only sees the Simulation through
   * Interface::Simulation and is thus independent of the hyperbolic and
   * the parabolic description.
   *
   * The tasks are constructed (and declare their run time parameters) in
   * the order of the subsections "/I - VTUOutput" to "/N - Checkpointing",
   * which is the order of their phases in a cycle.
   *
   * The Simulation and the Logger are required tasks; all other tasks
   * are selected with the run time parameter "tasks" of the Scheduler.
   *
   * @note This class sets up the default simulation chain of tasks used in
   * ryujin for any given Equation.
   *
   * @ingroup Driver
   */
  template <int dim, typename Number = double>
  struct TaskComposition {
    /**
     * Constructor taking the Simulation the tasks work with.
     */
    TaskComposition(Interface::Simulation<dim, Number> &simulation)
        : simulation(simulation)
        , vtu_output(simulation, /*levelsets*/ false, "/I - VTUOutput/full")
        , vtu_levelsets(
              simulation, /*levelsets*/ true, "/I - VTUOutput/levelsets")
        , quantities(simulation, "/J - Quantities")
        , error_evaluation(simulation, "/K - ErrorEvaluation")
        , mesh_adaptor(simulation, "/L - MeshAdaptor")
        , logger(simulation, "/M - Logger")
        , checkpointing(simulation, "/N - Checkpointing")
        , scheduler(simulation.mpi_ensemble().world_communicator(),
                    {&simulation, &logger},
                    {&vtu_output,
                     &vtu_levelsets,
                     &quantities,
                     &error_evaluation,
                     &mesh_adaptor,
                     &checkpointing})
    {
    }

    /**
     * Run the simulation.
     */
    void run()
    {
      scheduler.run();
    }

    Interface::Simulation<dim, Number> &simulation;

    VTUOutput<dim, Number> vtu_output;
    VTUOutput<dim, Number> vtu_levelsets;
    Quantities<dim, Number> quantities;
    ErrorEvaluation<dim, Number> error_evaluation;
    MeshAdaptor<dim, Number> mesh_adaptor;
    Logger<dim, Number> logger;
    Checkpointing<dim, Number> checkpointing;

    Scheduler<Number> scheduler;
  };
} // namespace ryujin
