//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/driver/environment.h>
#include <ryujin/driver/equation.h>

/*
 * A minimal executable using the ryujin library: run the registered
 * equation "euler" in two dimensions with the parameters of euler.prm.
 */
int main(int argc, char *argv[])
{
  ryujin::Environment environment(argc, argv);

  ryujin::equations().at("euler").run(
      2, "euler.prm", environment.mpi_communicator());

  return 0;
}
