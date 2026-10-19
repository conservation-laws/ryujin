//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>

#include <deal.II/base/mpi.h>

#include <string>

namespace ryujin
{
  namespace Interface
  {
    /**
     * An equation: a pairing of a hyperbolic and a parabolic description
     * that can be selected by name with the run time parameter "equation".
     *
     * @ingroup Interfaces
     */
    class Equation
    {
    public:
      virtual ~Equation() = default;

      /**
       * Construct and run the simulation for dimension @p dim with the
       * parameters of @p parameter_file.
       */
      virtual void run(int dim,
                       const std::string &parameter_file,
                       const MPI_Comm &mpi_comm) const = 0;

      /**
       * Write the default parameter files of the equation for all
       * dimensions.
       */
      virtual void create_parameter_files() const = 0;

      /**
       * The name of the equation, as selected by the run time parameter
       * "equation".
       */
      ACCESSOR_READ_ONLY(name)

    protected:
      Equation(const std::string &name)
          : name_(name)
      {
      }

    private:
      const std::string name_;
    };
  } // namespace Interface
} // namespace ryujin
