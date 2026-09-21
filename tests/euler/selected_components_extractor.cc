#include <description.h>

#include <discretization.h>
#include <mpi_ensemble.h>
#include <offline_data.h>
#include <selected_components_extractor.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <iomanip>
#include <iostream>
#include <sstream>

//
// Test the SelectedComponentsExtractor: We set up a small rectangular
// domain, populate a state vector with synthetic data and extract a
// selection of conserved, primitive, precomputed, and additional
// components. The extraction is performed on the host and on the default
// memory space and both results are printed.
//

using namespace ryujin;
using namespace dealii;

using Description = Euler::Description;
using HostSpace = dealii::MemorySpace::Host;
using DefaultSpace = dealii::MemorySpace::Default;

constexpr int dim = 2;
using Number = NUMBER;

using HyperbolicSystem = typename Description::HyperbolicSystem;
using ParabolicSystem = typename Description::ParabolicSystem;
using SystemView = typename HyperbolicSystem::template View<dim, Number>;

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

  typename SystemView::StateVector state_vector;
  auto &[U, precomputed, parabolic] = state_vector;
  U.reinit_with_scalar_partitioner(scalar_partitioner,
                                   TransferPolicy::implicit_transfers);
  precomputed.reinit_with_scalar_partitioner(
      scalar_partitioner, TransferPolicy::implicit_transfers);

  typename SystemView::InitialPrecomputedVector initial_precomputed;
  initial_precomputed.reinit_with_scalar_partitioner(
      scalar_partitioner, TransferPolicy::implicit_transfers);

  Vectors::ScalarVector<Number> alpha;
  alpha.reinit_with_scalar_partitioner(scalar_partitioner,
                                       TransferPolicy::implicit_transfers);

  {
    const auto U_view = U.view();
    const auto precomputed_view = precomputed.view();
    const auto alpha_view = alpha.view();

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

  /* Set up the extractor: */

  SelectedComponentsExtractor<Description, dim, Number> extractor(
      offline_data,
      hyperbolic_system,
      parabolic_system,
      initial_precomputed,
      {"alpha"},
      {alpha});

  const std::vector<std::string> selected{
      "rho", "m_1", "E", "v_1", "p", "s", "eta_h", "alpha"};
  extractor.prepare(selected);

  /* Extract on the host and on the default memory space: */

  const auto host_components =
      extractor.view<HostSpace>(state_vector).extract();
  const auto device_components =
      extractor.view<DefaultSpace>(state_vector).extract();

  Vectors::ScalarHostVector<Number> temp;
  temp.reinit(scalar_partitioner);

  std::cout << "n_locally_owned = " << n_owned << "\n";

  for (std::size_t k = 0; k < selected.size(); ++k) {
    temp.import_elements(device_components[k], VectorOperation::insert);

    std::cout << "\ncomponent \"" << selected[k] << "\":\n";
    for (unsigned int i = 0; i < n_owned; ++i) {
      std::cout << "  i = " << i                                     //
                << "  host: " << host_components[k].local_element(i) //
                << "  device: " << temp.local_element(i) << "\n";
    }
  }
}
