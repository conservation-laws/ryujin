#include <description.h>

#include <discretization.h>
#include <mpi_ensemble.h>
#include <offline_data.h>
#include <quantities.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/dofs/dof_tools.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

//
// Test the Quantities class: We set up a small rectangular domain with an
// interior (diagonal) and a boundary (left edge) manifold and select a
// conserved, two primitive, and a precomputed quantity. A state vector is
// populated with synthetic data that depends on the position of the
// degree of freedom and on time. We accumulate statistics at t = 0, ..., 4,
// write out all files, accumulate two more time steps (without clearing
// statistics), write out again, and print the resulting point maps,
// instantaneous values, time averaged central moments, and space averaged time
// series.
//

using namespace ryujin;
using namespace dealii;

using Description = Euler::Description;

constexpr int dim = 2;
using Number = NUMBER;

using HyperbolicSystem = typename Description::HyperbolicSystem;
using ParabolicSystem = typename Description::ParabolicSystem;
using SystemView = typename HyperbolicSystem::template View<dim, Number>;


/*
 * Populate the state vector with synthetic data for time step n: all
 * values are functions of the position of the degree of freedom (so that
 * the data is independent of the dof numbering) and are perturbed by a
 * factor f_n = sin(n * pi / 2), i.e., by 0, 1, 0, -1, 0, ...
 */
void populate(typename SystemView::StateVector &state_vector,
              const OfflineData<dim, Number> &offline_data,
              const std::map<types::global_dof_index, Point<dim>> &positions,
              const unsigned int n)
{
  static constexpr Number factors[4] = {0., 1., 0., -1.};
  const Number f = factors[n % 4];

  const auto &scalar_partitioner = offline_data.scalar_partitioner();
  const unsigned int n_owned = offline_data.n_locally_owned();

  auto &[U, precomputed, parabolic] = state_vector;
  const auto U_view = U.view();
  const auto precomputed_view = precomputed.view();

  for (unsigned int i = 0; i < n_owned; ++i) {
    const auto &position = positions.at(scalar_partitioner->local_to_global(i));
    const Number x = position[0];
    const Number y = position[1];

    typename SystemView::state_type U_i;
    U_i[0] = (1. + x + 2. * y) * (1. + 0.25 * f); /* rho */
    U_i[1] = 2. * x + 0.5 * f;                    /* m_1 */
    U_i[2] = y - 0.25;                            /* m_2 */
    U_i[dim + 1] = 4. + 2. * x + 2. * y + f;      /* E   */
    U_view.write_tensor(U_i, i);

    typename SystemView::precomputed_type precomputed_i;
    for (unsigned int d = 0; d < SystemView::n_precomputed_values; ++d)
      precomputed_i[d] = 100. * (d + 1) + 4. * x + 8. * y + f;
    precomputed_view.write_tensor(precomputed_i, i);
  }

  U_view.update_ghost_values();
  precomputed_view.update_ghost_values();
}


int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  /* Set up a small rectangular domain and the corresponding offline data: */

  MPIEnsemble mpi_ensemble(MPI_COMM_WORLD);

  HyperbolicSystem hyperbolic_system;
  ParabolicSystem parabolic_system;
  Discretization<dim> discretization(mpi_ensemble);
  OfflineData<dim, Number> offline_data(mpi_ensemble, discretization);

  typename SystemView::InitialPrecomputedVector initial_precomputed;

  Quantities<Description, dim, Number> quantities(mpi_ensemble,
                                                  offline_data,
                                                  hyperbolic_system,
                                                  parabolic_system,
                                                  initial_precomputed);

  std::stringstream parameters;
  parameters
      << "subsection Discretization\n"
      << "  set geometry        = rectangular domain\n"
      << "  set mesh refinement = 2\n"
      << "  subsection rectangular domain\n"
      << "    set position bottom left = 0, 0\n"
      << "    set position top right   = 1, 1\n"
      << "  end\n"
      << "end\n"
      << "subsection Quantities\n"
      << "  set quantities                   = rho, v_1, p, s\n"
      << "  set number of moments            = 4\n"
      << "  set clear statistics on writeout = false\n"
      << "  set interior manifolds           = "
      << "diagonal : x - y : instantaneous time_averaged space_averaged\n"
      << "  set boundary manifolds           = "
      << "left : x : instantaneous time_averaged space_averaged\n"
      << "end\n";
  ParameterAcceptor::initialize(parameters);

  discretization.prepare("test");
  offline_data.prepare(SystemView::problem_dimension,
                       SystemView::n_precomputed_values);

  const auto &scalar_partitioner = offline_data.scalar_partitioner();

  const auto positions = DoFTools::map_dofs_to_support_points(
      discretization.mapping(), offline_data.dof_handler());

  /*
   * We populate the state vector on the host and accumulate statistics on
   * the selected memory space, thus use an implicit transfer policy:
   */

  typename SystemView::StateVector state_vector;
  auto &[U, precomputed, parabolic] = state_vector;
  U.reinit_with_scalar_partitioner(scalar_partitioner,
                                   TransferPolicy::implicit_transfers);
  precomputed.reinit_with_scalar_partitioner(
      scalar_partitioner, TransferPolicy::implicit_transfers);
  initial_precomputed.reinit_with_scalar_partitioner(
      scalar_partitioner, TransferPolicy::implicit_transfers);

  /* Accumulate statistics and write out: */

  quantities.prepare("test");

  for (unsigned int n = 0; n <= 4; ++n) {
    populate(state_vector, offline_data, positions, n);
    quantities.accumulate(state_vector, Number(n));
  }
  quantities.write_out(state_vector, Number(4.), /*cycle*/ 0);

  for (unsigned int n = 5; n <= 6; ++n) {
    populate(state_vector, offline_data, positions, n);
    quantities.accumulate(state_vector, Number(n));
  }
  quantities.write_out(state_vector, Number(6.), /*cycle*/ 1);

  /* Print all output files: */

  if (mpi_ensemble.world_rank() != 0)
    return 0;

  for (const std::string manifold : {"diagonal", "left"})
    for (const std::string suffix : {"R0000-points",
                                     "R0001-instantaneous",
                                     "R0000-time_averaged",
                                     "R0001-time_averaged",
                                     "R0000-space_averaged_time_series"}) {
      const auto file_name = "test-" + manifold + "-" + suffix + ".dat";
      std::cout << "\n" << file_name << ":\n";
      std::ifstream file(file_name);
      std::cout << file.rdbuf();
    }
}
