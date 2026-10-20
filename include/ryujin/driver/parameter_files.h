//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/utilities.h>

#include <string>

namespace ryujin
{
  /**
   * Write the default parameter file "default_parameters-<name>.prm",
   * and, if @p write_detailed_description is set, the parameter file
   * "default_parameters-<name>-description.prm" with the documentation of
   * all parameters.
   *
   * The function calls @p setup that constructs the objects declaring the
   * run time parameters (typically a Simulation and a TaskComposition) and
   * returns them (for example in a std::unique_ptr). The objects are kept
   * alive while the parameter files are written and destroyed afterwards.
   * Finally, all parameters are cleared with
   * dealii::ParameterAcceptor::clear().
   *
   * @code
   * ryujin::write_parameter_files("euler-2d", true, [] {
   *   struct Objects {
   *     Simulation<HyperbolicDescription, ParabolicDescription, 2> simulation{
   *         MPI_COMM_SELF};
   *     TaskComposition<2> task_composition{simulation};
   *   };
   *   return std::make_unique<Objects>();
   * });
   * @endcode
   *
   * @ingroup Driver
   */
  template <typename Setup>
  void write_parameter_files(const std::string &name,
                             bool write_detailed_description,
                             const Setup &setup)
  {
    {
      auto &prm = dealii::ParameterAcceptor::prm;

      /*
       * Workaround: The Scheduler is constructed last and declares the
       * "A - Scheduler" section last. Add an entry to the section so that it
       * shows up first.
       */
      prm.enter_subsection("A - Scheduler");
      prm.declare_entry("basename", "test");
      prm.leave_subsection();

      [[maybe_unused]] const auto objects = setup();

      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_SELF) == 0) {
        if (write_detailed_description)
          prm.print_parameters(
              "default_parameters-" + name + "-description.prm",
              dealii::ParameterHandler::OutputStyle::KeepDeclarationOrder);

        prm.print_parameters(
            "default_parameters-" + name + ".prm",
            dealii::ParameterHandler::OutputStyle::Short |
                dealii::ParameterHandler::OutputStyle::KeepDeclarationOrder);
      }

      // all objects have to go out of scope, see
      // https://github.com/dealii/dealii/issues/15111
    }

    dealii::ParameterAcceptor::clear();
  }
} // namespace ryujin
