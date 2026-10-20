//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/driver/environment.h>
#include <ryujin/driver/equation.h>
#include <ryujin/driver/simulation.h>
#include <ryujin/driver/task_composition.h>
#include <ryujin/euler/hyperbolic_description.h>
#include <ryujin/stub/parabolic_module.h>

#include <deal.II/base/mpi.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

//
// Test write_parameter_files() with the default TaskComposition: the
// short and the detailed parameter file are written, and the subsection
// "A - Scheduler" comes first. We print all top-level subsections of the
// short parameter file.
//

using namespace ryujin;

using HyperbolicDescription = Euler::HyperbolicDescription;
using ParabolicDescription = StubParabolicDescription;
constexpr int dim = 2;
using Number = NUMBER;


int main(int argc, char *argv[])
{
  Environment environment(argc, argv);

  write_parameter_files("test", true, [] {
    struct Objects {
      Simulation<HyperbolicDescription, ParabolicDescription, dim, Number>
          simulation{MPI_COMM_SELF};
      TaskComposition<dim, Number> task_composition{simulation};
    };
    return std::make_unique<Objects>();
  });

  std::cout << std::boolalpha << "description written: "
            << std::filesystem::exists(
                   "default_parameters-test-description.prm")
            << std::endl;

  std::ifstream file("default_parameters-test.prm");
  std::string line;
  while (std::getline(file, line))
    if (line.starts_with("subsection "))
      std::cout << line << std::endl;

  return 0;
}
