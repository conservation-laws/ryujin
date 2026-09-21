#include <description.h>

#include <discretization.h>
#include <gpu.h>
#include <loop.h>
#include <mpi_ensemble.h>
#include <offline_data.h>
#include <selected_components_extractor.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <iomanip>
#include <iostream>
#include <sstream>

//
// Test SelectedComponentsExtractorView::extract_element(): We set up a
// small rectangular domain, populate a state vector with synthetic data
// and extract a selection of conserved, primitive, precomputed, and
// additional components for every degree of freedom individually. The
// loop is run on the host memory space (where it exercises both the
// vectorized and the scalar instantiation) and on the default memory
// space. Both results are printed and compared against the vectors
// returned by extract().
//
// Note that Euler has neither initial precomputed values, nor parabolic
// components. Those two code paths remain covered by the shallow water
// and navier stokes regression tests.
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

static const std::vector<std::string> selected{
    "rho", "m_1", "E", "v_1", "p", "s", "eta_h", "alpha"};

constexpr std::size_t n_selected = 8;


/*
 * Extract all selected components for every locally owned degree of
 * freedom with extract_element() and store the result in a
 * (component-wise) host vector.
 */
template <typename MemorySpace, typename Extractor, typename StateVector>
std::vector<Vectors::ScalarHostVector<Number>>
extract_element_wise(const Extractor &extractor,
                     const StateVector &state_vector,
                     const OfflineData<dim, Number> &offline_data)
{
  const unsigned int n_internal = offline_data.n_locally_internal();
  const unsigned int n_owned = offline_data.n_locally_owned();

  Mirrored<Number *> scratch("selected_components_extractor_02_scratch");
  scratch.reinit(n_selected * n_owned, TransferPolicy::implicit_transfers);

  extractor.template prepare_extraction<MemorySpace>(state_vector);
  const auto extractor_view = extractor.template view<MemorySpace>();
  auto *destination = scratch.template view<MemorySpace>();

  const auto body = [=](auto sentinel, unsigned int i) {
    using T = decltype(sentinel);

    T values[n_selected];
    extractor_view.extract_element(values, i);

    for (unsigned int k = 0; k < n_selected; ++k) {
      if constexpr (std::is_same_v<T, dealii::VectorizedArray<Number>>)
        values[k].store(destination + k * n_owned + i);
      else
        destination[k * n_owned + i] = values[k];
    }
  };

  loop<MemorySpace, Number>("extract_element", body, 0, n_internal, n_owned);

  /* Repackage the result in a vector of scalar host vectors: */

  const auto *result = std::as_const(scratch).template view<HostSpace>();

  std::vector<Vectors::ScalarHostVector<Number>> extracted(n_selected);
  for (unsigned int k = 0; k < n_selected; ++k) {
    extracted[k].reinit(offline_data.scalar_partitioner());
    for (unsigned int i = 0; i < n_owned; ++i)
      extracted[k].local_element(i) = result[k * n_owned + i];
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

  extractor.prepare(selected);

  AssertThrow(extractor.n_selected() == n_selected, dealii::ExcInternalError());

  /* Extract element wise on the host and on the default memory space: */

  const auto host_components =
      extract_element_wise<HostSpace>(extractor, state_vector, offline_data);
  const auto device_components =
      extract_element_wise<DefaultSpace>(extractor, state_vector, offline_data);

  /* And compare against the component wise extraction: */

  extractor.prepare_extraction<HostSpace>(state_vector);
  const auto reference = extractor.view<HostSpace>().extract();

  std::cout << "n_locally_owned = " << n_owned << "\n";

  for (std::size_t k = 0; k < n_selected; ++k) {
    std::cout << "\ncomponent \"" << selected[k] << "\":\n";

    Number host_difference = 0.;
    Number device_difference = 0.;

    for (unsigned int i = 0; i < n_owned; ++i) {
      const auto host = host_components[k].local_element(i);
      const auto device = device_components[k].local_element(i);

      std::cout << "  i = " << i      //
                << "  host: " << host //
                << "  device: " << device << "\n";

      host_difference = std::max(
          host_difference, std::abs(host - reference[k].local_element(i)));
      device_difference = std::max(
          device_difference, std::abs(device - reference[k].local_element(i)));
    }

    std::cout << "  difference to extract(): host: " << host_difference
              << "  device: " << device_difference << "\n";
  }
}
