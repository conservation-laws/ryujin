#include <limiter.h>
#include <multicomponent_vector.h>

#include <array>
#include <iomanip>
#include <iostream>

//
// Test that the LimiterView can be used on the device memory space: We
// compute bounds (projection_bounds_from_state(), combine_bounds(),
// fully_relax_bounds(), and the stencil-based reset(), accumulate(),
// bounds()) and run the convex limiter limit() over a set of states on the
// host and on the default memory space and print both results.
//

using namespace ryujin;

using HostSpace = dealii::MemorySpace::Host;
using DefaultSpace = dealii::MemorySpace::Default;

constexpr int dim = 2;
constexpr unsigned int problem_dimension = 2 + dim;
constexpr unsigned int n_states = 8;

using HostView = Euler::LimiterView<dim, double, HostSpace>;
using DeviceView = Euler::LimiterView<dim, double, DefaultSpace>;
using HostSystemView = Euler::HyperbolicSystemView<dim, double, HostSpace>;
using state_type = typename HostSystemView::state_type;


/*
 * Runtime parameters. These do not depend on a state and are thus computed
 * separately:
 */

constexpr const char *constant_names[]{"iterations",
                                       "newton_tolerance",
                                       "newton_max_iterations",
                                       "relaxation_factor"};

constexpr unsigned int n_constants = std::size(constant_names);


template <typename View>
DEAL_II_HOST_DEVICE dealii::Tensor<1, n_constants, double>
compute_constants(const View &view)
{
  dealii::Tensor<1, n_constants, double> result;
  unsigned int k = 0;

  result[k++] = view.iterations();
  result[k++] = view.newton_tolerance();
  result[k++] = view.newton_max_iterations();
  result[k++] = view.relaxation_factor();

  Assert(k == n_constants, dealii::ExcInternalError());
  return result;
}


/*
 * A list of all state-dependent quantities and their sizes - used for
 * printing the results:
 */

struct Quantity {
  const char *name;
  unsigned int size;
};

constexpr Quantity quantities[]{
    {"projection_bounds_from_state", 3},
    {"combine_bounds", 3},
    {"fully_relax_bounds", 3},
    {"bounds", 3},
    {"limit", 2},
};

constexpr unsigned int n_results = []() {
  unsigned int result = 0;
  for (const auto &quantity : quantities)
    result += quantity.size;
  return result;
}();


template <typename View, typename SystemView>
DEAL_II_HOST_DEVICE dealii::Tensor<1, n_results, double>
compute_quantities(const View &limiter_view,
                   const SystemView &system_view,
                   const typename View::PrecomputedVectorView &pv,
                   const typename SystemView::InitialPrecomputedVectorView &ipv,
                   const unsigned int i,
                   const state_type &U_i,
                   const unsigned int j_1,
                   const state_type &U_j_1,
                   const unsigned int j_2,
                   const state_type &U_j_2,
                   const dealii::Tensor<1, dim, double> &scaled_c_ij_1,
                   const dealii::Tensor<1, dim, double> &scaled_c_ij_2,
                   const double hd_i)
{
  using Bounds = typename View::Bounds;

  dealii::Tensor<1, n_results, double> result;
  unsigned int k = 0;

  const auto projection_bounds_i =
      limiter_view.projection_bounds_from_state(pv, i, U_i);
  for (unsigned int d = 0; d < 3; ++d)
    result[k++] = projection_bounds_i[d];

  const auto projection_bounds_j =
      limiter_view.projection_bounds_from_state(pv, j_1, U_j_1);
  const auto combined_bounds =
      limiter_view.combine_bounds(projection_bounds_i, projection_bounds_j);
  for (unsigned int d = 0; d < 3; ++d)
    result[k++] = combined_bounds[d];

  const auto relaxed_bounds =
      limiter_view.fully_relax_bounds(combined_bounds, hd_i);
  for (unsigned int d = 0; d < 3; ++d)
    result[k++] = relaxed_bounds[d];

  /* The view is stateful, work on a copy: */
  auto limiter = limiter_view;

  const state_type affine_shift; /* zero vector */
  const auto flux_i = system_view.flux_contribution(pv, ipv, i, U_i);
  const auto flux_j_1 = system_view.flux_contribution(pv, ipv, &j_1, U_j_1);
  const auto flux_j_2 = system_view.flux_contribution(pv, ipv, &j_2, U_j_2);

  limiter.reset(pv, i, U_i, flux_i);
  limiter.accumulate(pv, &j_1, U_j_1, flux_j_1, scaled_c_ij_1, affine_shift);
  limiter.accumulate(pv, &j_2, U_j_2, flux_j_2, scaled_c_ij_2, affine_shift);

  const auto accumulated_bounds = limiter.bounds(hd_i);
  for (unsigned int d = 0; d < 3; ++d)
    result[k++] = accumulated_bounds[d];

  /*
   * Limit against the tight combined projection bounds so that the
   * density limiter and the quadratic Newton solver actually engage:
   */
  const state_type P = 8. * (U_j_1 - U_i) + 4. * (U_j_2 - U_i);
  const auto [t_l, success] = limiter.limit(combined_bounds, U_i, P);
  result[k++] = t_l;
  result[k++] = success ? 1. : 0.;

  Assert(k == n_results, dealii::ExcInternalError());
  return result;
}


int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  std::cout << std::setprecision(10);
  std::cout << std::scientific;

  Euler::HyperbolicSystem hyperbolic_system;
  Euler::Limiter<double> limiter(hyperbolic_system);

  /* Exercise the parse_parameters_call_back() update path: */

  std::stringstream parameters;
  parameters << "subsection Limiter\n"
             << "set relaxation factor = 2.0\n"
             << "end" << std::endl;
  dealii::ParameterAcceptor::initialize(parameters);

  const auto host_system_view =
      hyperbolic_system.view<dim, double, HostSpace>();
  const auto device_system_view =
      hyperbolic_system.view<dim, double, DefaultSpace>();
  const auto host_view = limiter.view<dim, double, HostSpace>();
  const auto device_view = limiter.view<dim, double, DefaultSpace>();

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

  /* Note: the Euler equations have no precomputed initial values. */
  typename HostSystemView::InitialPrecomputedVector initial_precomputed;
  initial_precomputed.reinit_with_scalar_partitioner(scalar_partitioner);

  Vectors::MultiComponentVector<double, n_constants> constants;
  constants.reinit_with_scalar_partitioner(scalar_partitioner);

  Vectors::MultiComponentVector<double, n_results> results;
  results.reinit_with_scalar_partitioner(scalar_partitioner);

  /* Fill states and precomputed values on the host space: */

  for (unsigned int i = 0; i < n_states; ++i) {
    state_type primitive;
    primitive[0] = 1. + 0.125 * i;
    primitive[1] = 0.1 * i;
    primitive[2] = -0.05 * i;
    primitive[3] = 1. + 0.25 * i;
    const auto U_i = host_system_view.from_primitive_state(primitive);
    U.write_tensor(U_i, i);

    typename HostSystemView::precomputed_type prec_i;
    prec_i[0] = host_system_view.specific_entropy(U_i);
    prec_i[1] = host_system_view.harten_entropy(U_i);
    precomputed.write_tensor(prec_i, i);
  }

  /* Two scaled c_ij vectors and a scaled measure used for the stencil: */

  dealii::Tensor<1, dim, double> scaled_c_ij_1;
  scaled_c_ij_1[0] = 0.25;
  scaled_c_ij_1[1] = 0.5;

  dealii::Tensor<1, dim, double> scaled_c_ij_2;
  scaled_c_ij_2[0] = -0.5;
  scaled_c_ij_2[1] = 0.125;

  const double hd_i = 0.01;

  /* Compute all quantities on the host: */

  const auto host_constants = compute_constants(host_view);

  std::array<dealii::Tensor<1, n_results, double>, n_states> host_results;
  {
    const auto pv = precomputed.get_view<HostSpace>();
    const auto ipv = initial_precomputed.get_view<HostSpace>();

    for (unsigned int i = 0; i < n_states; ++i) {
      const unsigned int j_1 = (i + 1) % n_states;
      const unsigned int j_2 = (i + 7) % n_states;
      const auto U_i = U.read_tensor<double>(i);
      const auto U_j_1 = U.read_tensor<double>(j_1);
      const auto U_j_2 = U.read_tensor<double>(j_2);
      host_results[i] = compute_quantities(host_view,
                                           host_system_view,
                                           pv,
                                           ipv,
                                           i,
                                           U_i,
                                           j_1,
                                           U_j_1,
                                           j_2,
                                           U_j_2,
                                           scaled_c_ij_1,
                                           scaled_c_ij_2,
                                           hd_i);
    }
  }

  /* Compute the same quantities on the default space: */

  U.move_to_memory_space<DefaultSpace>();
  precomputed.move_to_memory_space<DefaultSpace>();
  initial_precomputed.move_to_memory_space<DefaultSpace>();
  constants.move_to_memory_space<DefaultSpace>();
  results.move_to_memory_space<DefaultSpace>();

  const auto U_view = U.get_view<DefaultSpace>();
  const auto pv = precomputed.get_view<DefaultSpace>();
  const auto ipv = initial_precomputed.get_view<DefaultSpace>();
  const auto constants_view = constants.get_view<DefaultSpace>();
  const auto results_view = results.get_view<DefaultSpace>();

  using ExecutionSpace = DefaultSpace::kokkos_space::execution_space;
  const auto exec = ExecutionSpace{};

  Kokkos::parallel_for(
      "test_constants",
      Kokkos::RangePolicy<ExecutionSpace>(exec, 0, 1),
      KOKKOS_LAMBDA(std::size_t i) {
        constants_view.write_tensor(compute_constants(device_view), i);
      });

  Kokkos::parallel_for(
      "test_quantities",
      Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n_states),
      KOKKOS_LAMBDA(std::size_t i) {
        const unsigned int j_1 = (i + 1) % n_states;
        const unsigned int j_2 = (i + 7) % n_states;
        const auto U_i = U_view.read_tensor(i);
        const auto U_j_1 = U_view.read_tensor(j_1);
        const auto U_j_2 = U_view.read_tensor(j_2);
        const auto result = compute_quantities(device_view,
                                               device_system_view,
                                               pv,
                                               ipv,
                                               i,
                                               U_i,
                                               j_1,
                                               U_j_1,
                                               j_2,
                                               U_j_2,
                                               scaled_c_ij_1,
                                               scaled_c_ij_2,
                                               hd_i);
        results_view.write_tensor(result, i);
      });

  constants.move_to_memory_space<HostSpace>();
  results.move_to_memory_space<HostSpace>();

  const unsigned int index = 0;
  const auto device_constants = constants.read_tensor<double>(index);

  std::array<dealii::Tensor<1, n_results, double>, n_states> device_results;
  for (unsigned int i = 0; i < n_states; ++i)
    device_results[i] = results.read_tensor<double>(i);

  /* Print all results: */

  std::cout << "Runtime parameters:\n\n";
  for (unsigned int k = 0; k < n_constants; ++k) {
    std::cout << constant_names[k] << " (host):   " << host_constants[k]
              << "\n";
    std::cout << constant_names[k] << " (device): " << device_constants[k]
              << "\n";
  }

  std::cout << "\nLimiter quantities for " << n_states << " states:\n";
  unsigned int offset = 0;
  for (const auto &quantity : quantities) {
    std::cout << "\n" << quantity.name << " (host):  ";
    for (unsigned int i = 0; i < n_states; ++i)
      for (unsigned int k = 0; k < quantity.size; ++k)
        std::cout << " " << host_results[i][offset + k];

    std::cout << "\n" << quantity.name << " (device):";
    for (unsigned int i = 0; i < n_states; ++i)
      for (unsigned int k = 0; k < quantity.size; ++k)
        std::cout << " " << device_results[i][offset + k];
    std::cout << "\n";

    offset += quantity.size;
  }
}
