#include <multicomponent_vector.h>
#include <wave_speed_estimator.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <sstream>

//
// Test that the WaveSpeedEstimatorView can be used on the device memory
// space: We compute the maximal wave speed estimate for a set of state
// pairs on the host and on the default memory space and print both
// results.
//
// We set nondefault runtime parameters and test, both, the guaranteed upper
// bound and the greedy wavespeed estimate. The flux and its gradient are
// evaluated on the host and stored as precomputed values. Enforcing
// entropy inequalities for additional Krŭzkov entropies requires calling
// into the selected flux and is thus not available on the device memory
// space.
//

using namespace ryujin;

using HostSpace = dealii::MemorySpace::Host;
using DefaultSpace = dealii::MemorySpace::Default;

constexpr int dim = 2;
constexpr unsigned int problem_dimension = 1;
constexpr unsigned int n_states = 8;

using HostView =
    ScalarConservation::WaveSpeedEstimatorView<dim, double, HostSpace>;
using DeviceView =
    ScalarConservation::WaveSpeedEstimatorView<dim, double, DefaultSpace>;
using HostSystemView =
    ScalarConservation::HyperbolicSystemView<dim, double, HostSpace>;
using state_type = typename HostSystemView::state_type;


constexpr const char *quantity_names[]{
    "use_greedy_wavespeed",
    "lambda_max",
    "lambda_max (precomputed values)",
    "use_greedy_wavespeed (greedy)",
    "lambda_max (greedy)",
    "lambda_max (greedy, precomputed values)"};
constexpr unsigned int n_results = std::size(quantity_names);


template <typename View, typename SystemView>
DEAL_II_HOST_DEVICE dealii::Tensor<1, n_results, double>
compute_quantities(const View &wave_speed_estimator_view,
                   const View &greedy_wave_speed_estimator_view,
                   const SystemView &system_view,
                   const typename View::PrecomputedVectorView &pv,
                   const unsigned int i,
                   const state_type &U_i,
                   const unsigned int j,
                   const state_type &U_j,
                   const dealii::Tensor<1, dim, double> &n_ij)
{
  using precomputed_type = typename View::precomputed_type;

  dealii::Tensor<1, n_results, double> result;
  unsigned int k = 0;

  const auto u_i = system_view.state(U_i);
  const auto u_j = system_view.state(U_j);
  const auto prec_i = pv.template read_tensor<double, precomputed_type>(i);
  const auto prec_j = pv.template read_tensor<double, precomputed_type>(&j);

  for (const auto &view :
       {wave_speed_estimator_view, greedy_wave_speed_estimator_view}) {
    result[k++] = view.use_greedy_wavespeed() ? 1. : 0.;

    result[k++] = view.compute(pv, U_i, U_j, i, &j, n_ij);

    /* Exercise the variant taking precomputed values directly: */
    result[k++] = view.compute(u_i, u_j, prec_i, prec_j, n_ij);
  }

  Assert(k == n_results, dealii::ExcInternalError());
  return result;
}


int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  std::cout << std::setprecision(10);
  std::cout << std::scientific;

  ScalarConservation::HyperbolicSystem hyperbolic_system;
  ScalarConservation::WaveSpeedEstimator<double> wave_speed_estimator(
      hyperbolic_system);
  ScalarConservation::WaveSpeedEstimator<double> greedy_wave_speed_estimator(
      hyperbolic_system, "/GreedyWaveSpeedEstimator");

  /* Exercise the parse_parameters_call_back() update path: */

  std::stringstream parameters;
  parameters << "subsection HyperbolicSystem\n"
             << "set flux = function\n"
             << "subsection function\n"
             << "set expression = 0.5*u*u;u*u*u/3\n"
             << "set derivative approximation delta = 1.0e-4\n"
             << "end\n"
             << "end\n"
             << "subsection GreedyWaveSpeedEstimator\n"
             << "set use greedy wavespeed = true\n"
             << "end" << std::endl;
  dealii::ParameterAcceptor::initialize(parameters);

  const auto host_system_view =
      hyperbolic_system.view<dim, double, HostSpace>();
  const auto device_system_view =
      hyperbolic_system.view<dim, double, DefaultSpace>();
  const auto host_view = wave_speed_estimator.view<dim, double, HostSpace>();
  const auto device_view =
      wave_speed_estimator.view<dim, double, DefaultSpace>();
  const auto greedy_host_view =
      greedy_wave_speed_estimator.view<dim, double, HostSpace>();
  const auto greedy_device_view =
      greedy_wave_speed_estimator.view<dim, double, DefaultSpace>();

  /* Set up locally owned and relevant index sets. */

  dealii::IndexSet locally_owned(n_states);
  dealii::IndexSet locally_relevant(n_states);
  locally_owned.add_range(0, n_states);
  locally_relevant.add_range(0, n_states);

  const auto scalar_partitioner =
      std::make_shared<dealii::Utilities::MPI::Partitioner>(
          locally_owned, locally_relevant, MPI_COMM_WORLD);

  Vectors::MultiComponentVector<double, problem_dimension> U;
  U.reinit_with_scalar_partitioner(scalar_partitioner);

  typename HostSystemView::PrecomputedVector precomputed;
  precomputed.reinit_with_scalar_partitioner(scalar_partitioner);

  Vectors::MultiComponentVector<double, n_results> results;
  results.reinit_with_scalar_partitioner(scalar_partitioner);

  /*
   * Fill states and precomputed values on the host space. The flux and
   * its gradient are computed by calling into the selected flux, which is
   * only possible on the host:
   */
  {
    const auto U_view = U.view();
    const auto precomputed_view = precomputed.view();

    for (unsigned int i = 0; i < n_states; ++i) {
      state_type U_i;
      U_i[0] = -1.0 + 0.35 * i * (i - 4.);
      U_view.write_tensor(U_i, i);

      const auto u_i = host_system_view.state(U_i);
      const auto f_i = host_system_view.flux_function(u_i);
      const auto df_i = host_system_view.flux_gradient_function(u_i);

      typename HostSystemView::precomputed_type prec_i;
      for (unsigned int d = 0; d < dim; ++d) {
        prec_i[d] = f_i[d];
        prec_i[dim + d] = df_i[d];
      }
      precomputed_view.write_tensor(prec_i, i);
    }
  }

  /* A normal used for the computations below: */

  dealii::Tensor<1, dim, double> normal;
  normal[0] = 0.6;
  normal[1] = -0.8;

  /* Compute all quantities on the host: */

  std::array<dealii::Tensor<1, n_results, double>, n_states> host_results;
  {
    const auto U_view = U.view<HostSpace>();
    const auto pv = precomputed.view<HostSpace>();

    for (unsigned int i = 0; i < n_states; ++i) {
      const unsigned int j = (i + 3) % n_states;
      const auto U_i = U_view.read_tensor<double>(i);
      const auto U_j = U_view.read_tensor<double>(j);
      host_results[i] = compute_quantities(host_view,
                                           greedy_host_view,
                                           host_system_view,
                                           pv,
                                           i,
                                           U_i,
                                           j,
                                           U_j,
                                           normal);
    }
  }

  /* Compute the same quantities on the default space: */

  U.move_to_memory_space<DefaultSpace>();
  precomputed.move_to_memory_space<DefaultSpace>();
  results.move_to_memory_space<DefaultSpace>();

  const auto U_view = U.view<DefaultSpace>();
  const auto pv = precomputed.view<DefaultSpace>();
  const auto results_view = results.view<DefaultSpace>();

  using ExecutionSpace = DefaultSpace::kokkos_space::execution_space;
  const auto exec = ExecutionSpace{};

  Kokkos::parallel_for(
      "test_quantities",
      Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n_states),
      KOKKOS_LAMBDA(std::size_t i) {
        const unsigned int j = (i + 3) % n_states;
        const auto U_i = U_view.read_tensor(i);
        const auto U_j = U_view.read_tensor(j);
        const auto result = compute_quantities(device_view,
                                               greedy_device_view,
                                               device_system_view,
                                               pv,
                                               i,
                                               U_i,
                                               j,
                                               U_j,
                                               normal);
        results_view.write_tensor(result, i);
      });

  results.move_to_memory_space<HostSpace>();

  std::array<dealii::Tensor<1, n_results, double>, n_states> device_results;
  {
    const auto results_host_view = results.view<HostSpace>();
    for (unsigned int i = 0; i < n_states; ++i)
      device_results[i] = results_host_view.read_tensor<double>(i);
  }

  /* Print all results: */

  std::cout << "Wave speed estimates for " << n_states << " states:\n";
  for (unsigned int k = 0; k < n_results; ++k) {
    std::cout << "\n" << quantity_names[k] << " (host):  ";
    for (unsigned int i = 0; i < n_states; ++i)
      std::cout << " " << host_results[i][k];

    std::cout << "\n" << quantity_names[k] << " (device):";
    for (unsigned int i = 0; i < n_states; ++i)
      std::cout << " " << device_results[i][k];
    std::cout << "\n";
  }
}
