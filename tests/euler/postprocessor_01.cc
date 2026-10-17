#include <hyperbolic_description.h>

#include <discretization.h>
#include <mpi_ensemble.h>
#include <offline_data.h>
#include <stub_parabolic_system.h>

#include <ryujin/typed/postprocessor.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

//
// Test the Postprocessor: We set up a small rectangular domain, populate
// a state vector with synthetic data and extract a selection of
// conserved, primitive, precomputed, and additional components. All components
// are extracted on the host with extract(), and at all locally owned degrees of
// freedom with the gather variant of extract() on the selected memory space.
// Both results are printed.
//

using namespace ryujin;
using namespace dealii;

using HyperbolicDescription = Euler::HyperbolicDescription;

constexpr int dim = 2;
using Number = NUMBER;

using HyperbolicSystem = typename HyperbolicDescription::HyperbolicSystem;
using ParabolicSystem = StubParabolicSystem;
using SystemView = typename HyperbolicSystem::template View<dim, Number>;


/*
 * Extract all selected components at the locally owned degrees of
 * freedom, visited in the given @p order, with the gather variant of
 * extract() and store the result in a (component-wise) host vector.
 */
std::vector<ScalarHostVector<Number>> extract_gather(
    const Interface::Postprocessor<dim, Number> &postprocessor,
    const Interface::Postprocessor<dim, Number>::Selection &selection,
    const StateVector<Number> &state_vector,
    const OfflineData<dim, Number> &offline_data,
    const std::vector<unsigned int> &order)
{
  const unsigned int n_selected = selection.size();
  const auto n_points = static_cast<unsigned int>(order.size());

  Mirrored<unsigned int *> indices("extract_gather_indices");
  indices.reinit(n_points, TransferPolicy::implicit_transfers);
  std::copy(order.begin(), order.end(), indices.view());

  Mirrored<Number *> values("extract_gather_values");
  values.reinit(n_selected * n_points, TransferPolicy::implicit_transfers);

  postprocessor.extract(
      selection,
      state_vector,
      std::as_const(indices).template view<selected_memory_space_t>(),
      n_points,
      values.template view<selected_memory_space_t>());

  /* Repackage the result in a vector of scalar host vectors: */

  const auto *result = std::as_const(values).view();

  std::vector<ScalarHostVector<Number>> extracted(n_selected);
  for (unsigned int k = 0; k < n_selected; ++k) {
    extracted[k].reinit(offline_data.scalar_partitioner());
    for (unsigned int p = 0; p < n_points; ++p)
      extracted[k].local_element(order[p]) = result[p * n_selected + k];
  }

  return extracted;
}

int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  std::cout << std::setprecision(10);
  std::cout << std::scientific;

  /* Set up a small rectangular domain and the corresponding offline data: */

  MPIEnsemble mpi_ensemble(MPI_COMM_WORLD);

  HyperbolicSystem hyperbolic_system;
  ParabolicSystem parabolic_system;
  Discretization<dim> discretization(mpi_ensemble);
  OfflineData<dim, Number> offline_data(mpi_ensemble, discretization);

  std::stringstream parameters;
  parameters << "subsection Discretization\n"
             << "  set geometry        = rectangular domain\n"
             << "  set mesh refinement = 1\n"
             << "end\n";
  ParameterAcceptor::initialize(parameters);

  discretization.prepare("test");
  offline_data.prepare(SystemView::problem_dimension,
                       SystemView::n_precomputed_values);

  const auto &scalar_partitioner = offline_data.scalar_partitioner();
  const unsigned int n_owned = offline_data.n_locally_owned();

  /*
   * Set up and populate a state vector, an additional scalar vector, and
   * an (empty) initial precomputed vector. We use an implicit transfer
   * policy so that we can create views on both memory spaces.
   */

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

  ScalarVector<Number> alpha;
  alpha.reinit_with_scalar_partitioner(
      scalar_partitioner, 1, TransferPolicy::implicit_transfers);

  {
    const auto U_view = U.view<SystemView::problem_dimension>();
    const auto precomputed_view = Q.view<SystemView::n_precomputed_values>();
    const auto alpha_view = alpha.view<1>();

    for (unsigned int i = 0; i < n_owned; ++i) {
      /* Use the global dof index so that the values are rank independent: */
      const auto index = Number(scalar_partitioner->local_to_global(i));

      typename SystemView::state_type U_i;
      U_i[0] = 1. + 0.125 * index;      /* rho */
      U_i[1] = 0.125 * index;           /* m_1 */
      U_i[2] = -0.25 + 0.0625 * index;  /* m_2 */
      U_i[dim + 1] = 4. + 0.25 * index; /* E   */
      U_view.write_tensor(U_i, i);

      typename SystemView::precomputed_type precomputed_i;
      for (unsigned int d = 0; d < SystemView::n_precomputed_values; ++d)
        precomputed_i[d] = 100. * (d + 1) + index;
      precomputed_view.write_tensor(precomputed_i, i);

      alpha_view.write_entry(0.5 * index, i);
    }

    U_view.update_ghost_values();
    precomputed_view.update_ghost_values();
    alpha_view.update_ghost_values();
  }

  /* Set up the postprocessor: */

  Postprocessor<HyperbolicDescription, dim, Number> postprocessor(
      offline_data,
      hyperbolic_system,
      parabolic_system,
      initial_precomputed,
      {"alpha"},
      {alpha});

  const std::vector<std::string> selected{
      "rho", "m_1", "E", "v_1", "p", "s", "eta_h", "alpha"};
  const auto selection = postprocessor.prepare(selected);

  /* Extract on the host and gather on the selected memory space: */

  const auto host_components = postprocessor.extract(selection, state_vector);

  std::vector<unsigned int> order(n_owned);
  std::iota(order.begin(), order.end(), 0u);
  const auto device_components = extract_gather(
      postprocessor, selection, state_vector, offline_data, order);

  std::cout << "n_locally_owned = " << n_owned << "\n";

  for (std::size_t k = 0; k < selected.size(); ++k) {
    std::cout << "\ncomponent \"" << selected[k] << "\":\n";
    for (unsigned int i = 0; i < n_owned; ++i) {
      std::cout << "  i = " << i                                     //
                << "  host: " << host_components[k].local_element(i) //
                << "  device: " << device_components[k].local_element(i)
                << "\n";
    }
  }
}
