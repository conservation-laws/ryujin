//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/discretization/discretization.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/euler/hyperbolic_description.h>
#include <ryujin/euler_poisson/parabolic_system.h>
#include <ryujin/typed/postprocessor.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/dofs/dof_tools.h>

#include <iostream>
#include <sstream>

//
// Test that the Postprocessor reads a dynamic state stored in the other
// (cG or dG) finite element space: For a cG and a dG ansatz we store a
// linear potential phi in the dG and cG space, respectively, and compare
// the extracted values against phi evaluated at the support points of the
// ansatz space.
//

using namespace ryujin;
using namespace dealii;

using HyperbolicDescription = Euler::HyperbolicDescription;

constexpr int dim = 2;
using Number = NUMBER;

using HyperbolicSystem = typename HyperbolicDescription::HyperbolicSystem;
using ParabolicSystem = EulerPoisson::ParabolicSystem;
using SystemView = typename HyperbolicSystem::template View<dim, Number>;


/*
 * Interpolate phi = 1 + 2 x + 3 y into the locally owned range of
 * @p vector that is set up for @p dof_handler.
 */
void interpolate(const Discretization<dim> &discretization,
                 const DoFHandler<dim> &dof_handler,
                 ScalarVector<Number> &vector)
{
  const auto view = vector.view<1>();
  const auto &partitioner = *vector.deal_ii_vector().get_partitioner();
  const auto &support_points = dof_handler.get_fe().get_unit_support_points();
  std::vector<dealii::types::global_dof_index> local_dof_indices;

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned())
      continue;

    local_dof_indices.resize(cell->get_fe().n_dofs_per_cell());
    cell->get_dof_indices(local_dof_indices);
    const auto &mapping = discretization.mapping()[cell->active_fe_index()];

    for (unsigned int j = 0; j < local_dof_indices.size(); ++j) {
      if (!partitioner.in_local_range(local_dof_indices[j]))
        continue;
      const auto x =
          mapping.transform_unit_to_real_cell(cell, support_points[j]);
      view.write_entry(Number(1. + 2. * x[0] + 3. * x[1]),
                       partitioner.global_to_local(local_dof_indices[j]));
    }
  }
}


void run(const std::string &ansatz)
{
  MPIEnsemble mpi_ensemble(MPI_COMM_WORLD);

  HyperbolicSystem hyperbolic_system;
  ParabolicSystem parabolic_system;
  Discretization<dim> discretization(mpi_ensemble);
  OfflineData<dim, Number> offline_data(mpi_ensemble, discretization);

  std::stringstream parameters;
  parameters << "subsection Discretization\n"
             << "  set finite element ansatz = " << ansatz << "\n"
             << "  set geometry              = rectangular domain\n"
             << "  set mesh refinement       = 2\n"
             << "end\n";
  ParameterAcceptor::initialize(parameters);

  discretization.prepare("test");
  offline_data.prepare(SystemView::problem_dimension,
                       SystemView::n_precomputed_values);

  const auto &scalar_partitioner = offline_data.scalar_partitioner();
  const unsigned int n_owned = offline_data.n_locally_owned();

  StateVector<Number> state_vector;
  auto &[U, Q, U_dynamic, Q_dynamic] = state_vector;
  U.reinit_with_scalar_partitioner(scalar_partitioner,
                                   SystemView::problem_dimension,
                                   TransferPolicy::implicit_transfers);
  Q.reinit_with_scalar_partitioner(scalar_partitioner,
                                   SystemView::n_precomputed_values,
                                   TransferPolicy::implicit_transfers);

  MultiComponentVector<Number> initial_precomputed;
  initial_precomputed.reinit_with_scalar_partitioner(
      scalar_partitioner,
      SystemView::n_initial_precomputed_values,
      TransferPolicy::implicit_transfers);

  /* Store phi in the other finite element space: */

  const auto &other_dof_handler = discretization.have_discontinuous_ansatz()
                                      ? offline_data.dof_handler_cg()
                                      : offline_data.dof_handler_dg();

  const auto other_partitioner = std::make_shared<Utilities::MPI::Partitioner>(
      other_dof_handler.locally_owned_dofs(),
      DoFTools::extract_locally_relevant_dofs(other_dof_handler),
      mpi_ensemble.ensemble_communicator());

  U_dynamic.resize(1);
  U_dynamic[0].reinit_with_scalar_partitioner(
      other_partitioner, 1, TransferPolicy::implicit_transfers);
  interpolate(discretization, other_dof_handler, U_dynamic[0]);

  /* The exact values at the support points of the ansatz space: */

  ScalarVector<Number> exact;
  exact.reinit_with_scalar_partitioner(
      scalar_partitioner, 1, TransferPolicy::implicit_transfers);
  interpolate(discretization, offline_data.dof_handler(), exact);

  Postprocessor<HyperbolicDescription, dim, Number> postprocessor(
      offline_data, hyperbolic_system, parabolic_system, initial_precomputed);

  const auto selection = postprocessor.prepare({"phi"});
  const auto components = postprocessor.extract(selection, state_vector);

  Number difference = 0.;
  for (unsigned int i = 0; i < n_owned; ++i)
    difference = std::max(difference,
                          std::abs(components[0].local_element(i) -
                                   exact.deal_ii_vector().local_element(i)));
  difference =
      Utilities::MPI::max(difference, mpi_ensemble.ensemble_communicator());

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << ansatz << ": phi exact up to round-off: "
              << (difference < 1.e-5 ? "yes" : "no") << std::endl;

  ParameterAcceptor::clear();
}


int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  run("cG Q1");
  run("dG Q1");
}
