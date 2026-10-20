//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/interface/equation.h>

#include <deal.II/base/mpi.h>

#include <string>

namespace ryujin
{
#ifndef DOXYGEN
  /* forward declaration */
  template <int dim, typename Number>
  struct TaskComposition;
#endif

  /**
   * The equation that pairs the hyperbolic description @p HD with the
   * parabolic description @p PD, for the number type @p Number.
   *
   * The equation runs the class template @p TaskComposition for the selected
   * dimension: it constructs a `Simulation<HD, PD, dim, Number>` from the
   * MPI communicator and a `TaskComposition<dim, Number>` from the
   * Simulation, initializes all run time parameters, and calls the run()
   * function of the TaskComposition. The default is the
   * ryujin::TaskComposition with all library tasks; a client registers an
   * Equation with a TaskComposition of its own to run a different
   * composition of tasks.
   *
   * @ingroup Typed
   */
  template <typename HD,
            typename PD,
            typename Number = double,
            template <int, typename> class TaskComposition =
                ryujin::TaskComposition>
  class Equation final : public Interface::Equation
  {
  public:
    /**
     * Constructor taking the name of the equation.
     */
    Equation(const std::string &name)
        : Interface::Equation(name)
    {
    }

    void run(int dim,
             const std::string &parameter_file,
             const MPI_Comm &mpi_comm) const override;

    void create_parameter_files() const override;

  private:
    /**
     * Construct the Simulation and the TaskComposition for dimension
     * @p dim, initialize all parameters from @p parameter_file, and run.
     */
    template <int dim>
    static void run_simulation(const std::string &parameter_file,
                               const MPI_Comm &mpi_comm);

    /**
     * Write the default parameter file(s) for dimension @p dim.
     */
    template <int dim>
    void create_parameter_file(bool write_detailed_description) const;
  };
} // namespace ryujin
