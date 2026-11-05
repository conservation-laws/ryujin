//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/discretization/discretization.h>
#include <ryujin/discretization/offline_data.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

//
// Test the boundary map in 3D with hanging nodes on the boundary. The
// transformation turns the unit cube into a house with a roof whose ridge
// runs along x = 0.5. Only the cell on the right of the ridge is refined,
// so that the midpoints of the coarse edges at x = 0.5 are hanging nodes.
//
// Let A be the area of one side of the roof. A ridge vertex has the boundary
// mass A/4 on the left and, including its share of the hanging node, A/8 on
// the right. With the roof normals (-+1, 0, 2) / sqrt(5) this gives the
// normal (-1, 0, 6) / sqrt(37). The boundary map has to sum up to the
// surface of the domain and a vanishing normal, and the result must not
// depend on the number of MPI ranks.
//

using namespace ryujin;
using namespace dealii;

constexpr int dim = 3;
using Entry = std::array<double, 10>; // position, id, normal, masses

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  MPIEnsemble mpi_ensemble(MPI_COMM_WORLD);
  Discretization<dim> discretization(mpi_ensemble);
  OfflineData<dim, dim, NUMBER> offline_data(mpi_ensemble, discretization);

  std::stringstream parameters;
  parameters << "subsection Discretization\n"
             << "  set geometry                     = rectangular domain\n"
             << "  set mesh refinement              = 0\n"
             << "  set mesh local refinement        = 1\n"
             << "  set mesh local refinement region = x - 0.5\n"
             << "  subsection rectangular domain\n"
             << "    set position bottom left     = 0, 0, 0\n"
             << "    set position top right       = 1, 1, 1\n"
             << "    set subdivisions x           = 2\n"
             << "    set transformation           = x; y; z + "
                "0.25*z*(1-2*abs(x-0.5))\n"
             << "    set boundary condition left  = slip\n"
             << "    set boundary condition right = slip\n"
             << "    set boundary condition bottom = slip\n"
             << "    set boundary condition top   = slip\n"
             << "    set boundary condition back  = slip\n"
             << "    set boundary condition front = slip\n"
             << "  end\n"
             << "end\n";
  ParameterAcceptor::initialize(parameters);

  discretization.prepare("test");
  offline_data.prepare(/*problem dimension*/ 1, /*precomputed values*/ 1);

  std::vector<Entry> local_entries;
  for (const auto &[i, normal, normal_mass, boundary_mass, id, position] :
       offline_data.boundary_map()) {
    (void)i;
    local_entries.push_back({position[0],
                             position[1],
                             position[2],
                             double(id),
                             normal[0],
                             normal[1],
                             normal[2],
                             double(normal_mass),
                             double(boundary_mass),
                             0.});
  }

  const auto gathered = Utilities::MPI::gather(MPI_COMM_WORLD, local_entries);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0)
    return 0;

  std::vector<Entry> entries;
  for (const auto &part : gathered)
    entries.insert(entries.end(), part.begin(), part.end());

  double surface = 0.;
  Tensor<1, dim> closure;
  for (const auto &entry : entries) {
    surface += entry[8];
    for (unsigned int d = 0; d < dim; ++d)
      closure[d] += entry[7] * entry[4 + d];
  }

  /* Sort by position and normal with some tolerance: */
  for (auto &entry : entries)
    for (auto &value : entry)
      value = std::round(value * 1.e6) * 1.e-6 + 0.;
  std::sort(entries.begin(), entries.end());

  std::cout << std::fixed << std::setprecision(6);
  for (const auto &entry : entries)
    for (unsigned int k = 0; k < 9; ++k)
      std::cout << std::setw(10) << entry[k] << (k == 8 ? "\n" : " ");

  std::cout << "surface: " << surface << std::endl;
  std::cout << "closure: " << closure.norm() << std::endl;
}
