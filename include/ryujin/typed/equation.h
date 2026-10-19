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
  /**
   * The equation that pairs the @p HyperbolicDescription with the
   * @p ParabolicDescription, for the number type @p Number.
   *
   * @ingroup Typed
   */
  template <typename HyperbolicDescription,
            typename ParabolicDescription,
            typename Number = double>
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
     * Construct the Simulation and the TimeLoop for dimension @p dim,
     * initialize all parameters from @p parameter_file, and run.
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
