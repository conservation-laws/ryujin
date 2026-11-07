//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/discretization/discretization.h>
#include <ryujin/discretization/offline_data.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <iostream>
#include <sstream>

//
// Test that OfflineData::update_geometry() throws on all MPI ranks if the
// transformation inverts a part of the mesh. The fold at y = 1 inverts the
// lower half of the cells and leaves the upper half intact.
//
// In debug mode deal.II already asserts on distorted cells inside of the
// mapping, so this test only runs in release mode.
//

using namespace ryujin;
using namespace dealii;

constexpr int dim = 2;

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  MPIEnsemble mpi_ensemble(MPI_COMM_WORLD);
  Discretization<dim> discretization(mpi_ensemble);
  OfflineData<dim, dim, NUMBER> offline_data(mpi_ensemble, discretization);

  std::stringstream parameters;
  parameters << "subsection Discretization\n"
             << "  set geometry = rectangular domain\n"
             << "  set mesh refinement = 2\n"
             << "  subsection rectangular domain\n"
             << "    set position top right = 1, 2\n"
             << "    set subdivisions y = 2\n"
             << "  end\n"
             << "end\n";
  ParameterAcceptor::initialize(parameters);

  discretization.prepare("test");
  offline_data.prepare(/*problem dimension*/ 1, /*precomputed values*/ 1);

  discretization.transformation() = [](const auto &,
                                       const Point<dim> &x) -> Point<dim> {
    return {x[0], std::abs(x[1] - 1.)};
  };
  discretization.update_mapping();

  unsigned int n_thrown = 0;
  try {
    offline_data.update_geometry();
  } catch (const ExcMessage &) {
    n_thrown = 1;
  }

  n_thrown = Utilities::MPI::sum(n_thrown, MPI_COMM_WORLD);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "ranks that have thrown: " << n_thrown << std::endl;
}
