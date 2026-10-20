//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <deal.II/base/mpi.h>

#include <optional>

namespace ryujin
{
  /**
   * The execution environment of a program using ryujin.
   *
   * The constructor performs various tasks including
   *  - flushing denormals to zero,
   *  - initializing MPI,
   *  - setting up thread pools (obeying thread limits),
   *  - initializing instrumentation (LIKWID and NVTX markers).
   *
   *  The destructor closes the instrumentation and finalizes MPI.
   *
   * @code
   * int main(int argc, char *argv[])
   * {
   *   ryujin::Environment environment(argc, argv);
   *
   *   ryujin::Simulation<HyperbolicDescription, ParabolicDescription, dim>
   *       simulation(environment.mpi_communicator());
   *
   *   ryujin::TaskComposition<2> task_composition(simulation);
   *
   *   dealii::ParameterAcceptor::initialize("simulation.prm");
   *
   *   task_composition.run();
   * }
   * @endcode
   *
   * @ingroup Driver
   */
  class Environment final
  {
  public:
    /**
     * Constructor taking the command line arguments of the program.
     */
    Environment(int &argc, char **&argv);

    Environment(const Environment &) = delete;
    Environment &operator=(const Environment &) = delete;

    /**
     * Destructor.
     */
    ~Environment();

    /**
     * The MPI communicator.
     */
    MPI_Comm mpi_communicator() const
    {
      return MPI_COMM_WORLD;
    }

  private:
    std::optional<dealii::Utilities::MPI::MPI_InitFinalize> mpi_initialization_;
  };
} // namespace ryujin
