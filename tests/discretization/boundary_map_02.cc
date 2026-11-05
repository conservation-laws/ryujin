//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/discretization/discretization.h>
#include <ryujin/discretization/offline_data.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <iomanip>
#include <iostream>
#include <sstream>

//
// Test the boundary map for two more kinds of constraints: The first run
// uses the geometry of boundary_map_01 with Q2 elements, where constraint
// weights can be negative. The second run uses a two-dimensional channel with
// periodic left and right boundaries, where the walls end in constrained
// degrees of freedom. In both cases the boundary map has to sum up to the
// surface of the non-periodic boundary, and a negative weight must not create
// an additional boundary map entry.
//

using namespace ryujin;
using namespace dealii;

template <int dim>
void test(const std::string &parameter_string)
{
  MPIEnsemble mpi_ensemble(MPI_COMM_WORLD);
  Discretization<dim> discretization(mpi_ensemble);
  OfflineData<dim, dim, NUMBER> offline_data(mpi_ensemble, discretization);

  std::stringstream parameters(parameter_string);
  ParameterAcceptor::initialize(parameters);

  discretization.prepare("test");
  offline_data.prepare(/*problem dimension*/ 1, /*precomputed values*/ 1);

  double surface = 0.;
  Tensor<1, dim> closure;
  for (const auto &[i, normal, normal_mass, boundary_mass, id, position] :
       offline_data.boundary_map()) {
    (void)i, (void)id, (void)position;
    surface += boundary_mass;
    closure += normal_mass * Tensor<1, dim>(normal);
  }

  const auto n_entries = Utilities::MPI::sum(
      (unsigned int)offline_data.boundary_map().size(), MPI_COMM_WORLD);
  surface = Utilities::MPI::sum(surface, MPI_COMM_WORLD);
  closure = Utilities::MPI::sum(closure, MPI_COMM_WORLD);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "entries: " << n_entries << "\n"
              << "surface: " << surface << "\n"
              << "closure: " << closure.norm() << std::endl;

  ParameterAcceptor::clear();
}

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  std::cout << std::fixed << std::setprecision(6);

  test<3>("subsection Discretization\n"
          "  set finite element ansatz          = cG Q2\n"
          "  set geometry                       = rectangular domain\n"
          "  set mesh refinement                = 0\n"
          "  set mesh local refinement          = 1\n"
          "  set mesh local refinement region   = x - 0.5\n"
          "  subsection rectangular domain\n"
          "    set position bottom left         = 0, 0, 0\n"
          "    set position top right           = 1, 1, 1\n"
          "    set subdivisions x               = 2\n"
          "    set transformation               = x; y; "
          "z + 0.25*z*(1-2*abs(x-0.5))\n"
          "    set boundary condition left      = slip\n"
          "    set boundary condition right     = slip\n"
          "    set boundary condition bottom    = slip\n"
          "    set boundary condition top       = slip\n"
          "    set boundary condition back      = slip\n"
          "    set boundary condition front     = slip\n"
          "  end\n"
          "end\n");

  test<2>("subsection Discretization\n"
          "  set geometry                       = rectangular domain\n"
          "  set mesh refinement                = 1\n"
          "  subsection rectangular domain\n"
          "    set position bottom left         = 0, 0\n"
          "    set position top right           = 1, 1\n"
          "    set boundary condition left      = periodic\n"
          "    set boundary condition right     = periodic\n"
          "    set boundary condition bottom    = slip\n"
          "    set boundary condition top       = slip\n"
          "  end\n"
          "end\n");
}
