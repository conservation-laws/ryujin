//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#include <ryujin/base/compile_time_options.h>
#include <ryujin/driver/environment.h>
#include <ryujin/driver/equation.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/utilities.h>

#include <filesystem>
#include <iostream>
#include <string>

namespace
{
  /**
   * The Dave error message
   */
  const std::string dave =
      "\nDave, this conversation can serve no purpose anymore. Goodbye.\n\n";


  /**
   * The run time parameters "dimension" and "equation" that select the
   * equation to run.
   *
   * When starting up the ryujin executable we are faced with the following
   * difficulties:
   *  - The Simulation class is templated in the HyperbolicDescription, the
   *    ParabolicDescription, and the dimension that have to be read from the
   *    `ryujin.prm` parameter file.
   *  - The final set of valid parameters that can be configured in the
   *    `ryujin.prm` depend on the runtime parameters "dimension" and
   *    "equation" themselves.
   *
   * We thus first read in two parameters from the parameter file:
   * ```
   * subsection B - Equation
   *   set dimension           = ...
   *   set equation            = ...
   * end
   * ```
   * and then run the equation with the given name for the given dimension.
   *
   * @note The object has to stay alive while the equation runs: the
   * equation parses the complete parameter file again with
   * dealii::ParameterAcceptor::initialize(), which requires the two
   * parameters to be declared.
   */
  struct EquationSelection final : dealii::ParameterAcceptor {
    EquationSelection()
        : ParameterAcceptor("B - Equation")
    {
      add_parameter("dimension", dimension, "The spatial dimension");

      std::string names;
      for (const auto &[name, equation] : ryujin::equations())
        names += (names.empty() ? "" : "|") + name;

      equation = ryujin::equations().begin()->first;
      add_parameter("equation",
                    equation,
                    "The PDE system",
                    this->prm,
                    dealii::Patterns::Selection(names));
    }

    int dimension = 0;
    std::string equation;
  };
} // namespace


int main(int argc, char *argv[])
{
  /*
   * Entry point into the ryujin executable. We first determine what
   * parameter file to use, then read in the parameters "dimension" and
   * "equation", and run the selected equation.
   */

  ryujin::Environment environment(argc, argv);
  const MPI_Comm mpi_communicator = environment.mpi_communicator();
  const bool is_root =
      dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0;

  if (is_root) {
    std::cout << "[INFO] initiating flux capacitor" << std::endl;
  }

  if (argc > 2) {
    if (is_root) {
      std::cout << "[ERROR] Invalid number of parameters. At most one argument "
                << "supported which has to be a parameter file." << std::endl;
    }
    return 1;
  }

  const auto executable_name = std::filesystem::path(argv[0]).filename();
  std::string parameter_file = executable_name.string() + ".prm";

  if (argc == 2) {
    parameter_file = argv[1];

    if (!std::filesystem::exists(parameter_file)) {
      if (is_root) {
        std::cout << "[ERROR] The specified parameter file »" << parameter_file
                  << "« does not exist." << std::endl;
      }
      return 1;
    }
  }

  if (!std::filesystem::exists(parameter_file)) {
    if (is_root) {
      std::cout << "[INFO] Default parameter file »" << parameter_file
                << "« not found.\n[INFO] Creating template parameter files for "
                << "you. Please modify and rename one of the templates to »"
                << parameter_file << "«." << std::endl;
      for (const auto &[name, equation] : ryujin::equations())
        equation.create_parameter_files();
    }

    MPI_Barrier(mpi_communicator);
    return 1;
  }

  EquationSelection selection;
  dealii::ParameterAcceptor::prm.parse_input(
      parameter_file,
      "",
      /* skip undefined */ true,
      /* assert entries present */ false);

  AssertThrow(selection.dimension >= 1 && selection.dimension <= 3,
              dealii::ExcMessage(dave +
                                 "The dimension parameter needs to be "
                                 "either 1, 2, or 3, but we encountered »" +
                                 std::to_string(selection.dimension) + "«\n"));

  if (is_root) {
    std::cout << "[INFO] dispatching to driver »" << selection.equation
              << "« with dim=" << selection.dimension << std::endl;
  }

  ryujin::equations()
      .at(selection.equation)
      .run(selection.dimension, parameter_file, mpi_communicator);

  return 0;
}
