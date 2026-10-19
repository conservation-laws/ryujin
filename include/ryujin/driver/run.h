//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

namespace ryujin
{
  /**
   * The entry point of the ryujin executable.
   *
   * Initializes MPI and the thread pools, and determines the parameter
   * file: the only command line argument @p argv[1] if given, and
   * `<executable name>.prm` otherwise. If the parameter file does not
   * exist the default parameter files of all registered equations are
   * written. Otherwise the equation selected in the parameter file is run.
   *
   * A client executable registers its own equations, initial states, and
   * geometries with the respective Library and then calls this function:
   * @code
   * int main(int argc, char *argv[])
   * {
   *   ryujin::EquationLibrary::add(...);
   *   return ryujin::run(argc, argv);
   * }
   * @endcode
   *
   * Returns the exit code of the executable.
   *
   * @ingroup Driver
   */
  int run(int argc, char *argv[]);
} // namespace ryujin
