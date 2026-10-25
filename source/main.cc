//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#include <ryujin/base/compile_time_options.h>
#include <ryujin/driver/environment.h>
#include <ryujin/driver/equation.h>
#include <ryujin/driver/parameter_editor.h>
#include <ryujin/driver/splash_screen.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/utilities.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

namespace
{
  /**
   * A small struct to query the run time parameters "dimension" and
   * "equation" that select the equation to run.
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
   */
  struct EquationSelection final : dealii::ParameterAcceptor {
    EquationSelection()
        : ParameterAcceptor("B - Equation")
    {
      add_parameter("dimension",
                    dimension,
                    "The spatial dimension",
                    this->prm,
                    dealii::Patterns::Integer(1, 3));

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

    int dimension = 1;
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

  /*
   * Remove --edit, --editor, and --splash=<name> from the argument list
   * before handing off to MPI, Kokkos, PETSc, etc...
   */
  bool invoke_editor = false;
  auto splash_screen = ryujin::SplashScreen::logo;
  bool invalid_splash_screen = false;
  const auto end = std::remove_if(argv + 1, argv + argc, [&](const char *arg) {
    const std::string_view option = arg;
    if (option == "--edit" || option == "--editor")
      invoke_editor = true;
    else if (option == "--splash=none")
      splash_screen = ryujin::SplashScreen::none;
    else if (option == "--splash=logo")
      splash_screen = ryujin::SplashScreen::logo;
    else if (option == "--splash=arcade")
      splash_screen = ryujin::SplashScreen::arcade;
    else if (option.starts_with("--splash="))
      invalid_splash_screen = true;
    else
      return false;
    return true;
  });
  argc = static_cast<int>(end - argv);

  ryujin::Environment environment(argc, argv);
  const MPI_Comm mpi_communicator = environment.mpi_communicator();
  const bool is_root =
      dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0;

  if (is_root) {
    std::cout << "[INFO] initiating flux capacitor" << std::endl;
  }

  if (argc > 2) {
    if (is_root) {
      std::cerr << "[ERROR] Invalid number of parameters. At most one argument "
                << "supported which has to be a parameter file (and the "
                << "options --edit or --editor, and --splash=<name>)."
                << std::endl;
    }
    return 1;
  }

  if (invalid_splash_screen) {
    if (is_root) {
      std::cerr << "[ERROR] Invalid splash screen. Supported are "
                << "--splash=none, --splash=logo, and --splash=arcade."
                << std::endl;
    }
    return 1;
  }

  const bool default_file = argc == 1;
  const std::string parameter_file =
      default_file ? std::filesystem::path(argv[0]).filename().string() + ".prm"
                   : std::string(argv[1]);
  const bool file_exists = std::filesystem::exists(parameter_file);

  if (!file_exists && !default_file && !invoke_editor) {
    if (is_root) {
      std::cerr << "[ERROR] The specified parameter file »" << parameter_file
                << "« does not exist." << std::endl;
    }
    return 1;
  }

  /*
   * Open the parameter editor if requested, or if there is no parameter
   * file and we are running on an interactive terminal:
   */

  const bool editor_available =
      ryujin::parameter_editor_available(mpi_communicator);

  if (invoke_editor && !editor_available) {
    if (is_root) {
      std::cerr << "[ERROR] The parameter editor requires an interactive "
                << "terminal, a serial run, and ryujin compiled with FTXUI."
                << std::endl;
    }
    return 1;
  }

  if (invoke_editor || (editor_available && !file_exists)) {
    ryujin::edit_parameter_file(parameter_file);
    return 0;
  }

  if (!file_exists) {
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

  if (is_root) {
    std::cout << "[INFO] dispatching to driver »" << selection.equation
              << "« with dim=" << selection.dimension << std::endl;
  }

#if defined(WITH_FTXUI)
  if (is_root)
    ryujin::play_splash_screen(
        selection.equation,
        selection.dimension,
        dealii::Utilities::MPI::n_mpi_processes(mpi_communicator),
        splash_screen);
#else
  (void)splash_screen;
#endif

  ryujin::equations()
      .at(selection.equation)
      .run(selection.dimension, parameter_file, mpi_communicator);

  return 0;
}
