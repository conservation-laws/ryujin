//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/discretization/discretization.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/fe/fe_values.h>

#include <iostream>
#include <sstream>

//
// Test Discretization::transformation(): the geometry of a rectangular
// domain is changed by setting a transformation with constant Jacobian
// determinant 2, which doubles the area computed with the mapping. The
// transformation persists through a mesh adaptation cycle without marked
// cells, and an empty transformation restores the identity.
//

using namespace ryujin;
using namespace dealii;

constexpr int dim = 2;

double area(const Discretization<dim> &discretization,
            const MPIEnsemble &mpi_ensemble)
{
  FEValues<dim> fe_values(discretization.mapping()[0],
                          discretization.finite_element()[0],
                          discretization.quadrature()[0],
                          update_JxW_values);

  double result = 0.;
  for (const auto &cell :
       discretization.triangulation().active_cell_iterators()) {
    if (!cell->is_locally_owned())
      continue;
    fe_values.reinit(cell);
    for (const auto q : fe_values.quadrature_point_indices())
      result += fe_values.JxW(q);
  }

  return Utilities::MPI::sum(result, mpi_ensemble.ensemble_communicator());
}

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  MPIEnsemble mpi_ensemble(MPI_COMM_WORLD);
  Discretization<dim> discretization(mpi_ensemble);

  const auto print = [&](const std::string &label) {
    const auto result = area(discretization, mpi_ensemble);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << label << result << std::endl;
  };

  std::stringstream parameters;
  parameters << "subsection Discretization\n"
             << "  set geometry        = rectangular domain\n"
             << "  set mesh refinement = 2\n"
             << "end\n";
  ParameterAcceptor::initialize(parameters);

  discretization.prepare("test");
  discretization.update_mapping();
  print("geometry:              ");

  discretization.transformation() =
      [](const typename Triangulation<dim>::cell_iterator &,
         const Point<dim> &point) {
        return Point<dim>(2. * point[0], point[1] + 0.5 * point[0]);
      };
  discretization.update_mapping();
  print("transformation:        ");

  auto &triangulation = discretization.triangulation();
  triangulation.prepare_coarsening_and_refinement();
  triangulation.execute_coarsening_and_refinement();
  discretization.update_mapping();
  print("adaptation cycle:      ");

  discretization.transformation() = nullptr;
  discretization.update_mapping();
  print("empty transformation:  ");
}
