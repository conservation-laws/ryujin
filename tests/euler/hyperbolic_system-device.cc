#include <hyperbolic_system.h>
#include <multicomponent_vector.h>

#include <array>
#include <iomanip>
#include <iostream>

//
// Test that the HyperbolicSystemView can be used on the device memory
// space: We compute all runtime parameters and derived quantities of the
// view on the host and on the default memory space and print both results.
//
// Not covered by this test:
//  - fill_precomputed_values(): host only, this function drives a
//    cpu_simd_loop() over OfflineData.
//  - high_order_flux_divergence() and nodal_source(): deleted for the Euler
//    equations.
//  - component_names, primitive_component_names, precomputed_names: arrays
//    of std::string that are host only by nature.
//

using namespace ryujin;

using HostSpace = dealii::MemorySpace::Host;
using DefaultSpace = dealii::MemorySpace::Default;

constexpr int dim = 2;
constexpr unsigned int problem_dimension = 2 + dim;
constexpr unsigned int n_states = 8;

using HostView = Euler::HyperbolicSystemView<dim, double, HostSpace>;
using DeviceView = Euler::HyperbolicSystemView<dim, double, DefaultSpace>;
using state_type = typename HostView::state_type;


/*
 * Runtime parameters and cached inverses. These do not depend on a state
 * and are thus computed separately:
 */

constexpr const char *constant_names[]{"gamma",
                                       "reference_density",
                                       "vacuum_state_relaxation_small",
                                       "vacuum_state_relaxation_large",
                                       "gamma_inverse",
                                       "gamma_plus_one_inverse",
                                       "gamma_minus_one_inverse",
                                       "gamma_minus_one_over_gamma_plus_one"};

constexpr unsigned int n_constants = std::size(constant_names);


template <typename View>
DEAL_II_HOST_DEVICE dealii::Tensor<1, n_constants, double>
compute_constants(const View &view)
{
  dealii::Tensor<1, n_constants, double> result;
  unsigned int k = 0;

  result[k++] = view.gamma();
  result[k++] = view.reference_density();
  result[k++] = view.vacuum_state_relaxation_small();
  result[k++] = view.vacuum_state_relaxation_large();
  result[k++] = view.gamma_inverse();
  result[k++] = view.gamma_plus_one_inverse();
  result[k++] = view.gamma_minus_one_inverse();
  result[k++] = view.gamma_minus_one_over_gamma_plus_one();

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
    {"density", 1},
    {"filter_vacuum_density", 1},
    {"momentum", dim},
    {"total_energy", 1},
    {"internal_energy", 1},
    {"internal_energy_derivative", problem_dimension},
    {"pressure", 1},
    {"speed_of_sound", 1},
    {"specific_entropy", 1},
    {"harten_entropy", 1},
    {"harten_entropy_derivative", problem_dimension},
    {"mathematical_entropy", 1},
    {"mathematical_entropy_derivative", problem_dimension},
    {"is_admissible", 1},
    {"f", problem_dimension *dim},
    {"flux_divergence", problem_dimension},
    {"to_primitive_state", problem_dimension},
    {"from_primitive_state", problem_dimension},
    {"from_initial_state", problem_dimension},
    {"apply_galilei_transform", problem_dimension},
    {"linearized_eigenvector<1>", 2 * problem_dimension},
    {"linearized_eigenvector<problem_dimension>", 2 * problem_dimension},
    {"prescribe_riemann_characteristic<1>", problem_dimension},
    {"prescribe_riemann_characteristic<2>", problem_dimension},
    {"apply_boundary_conditions<dirichlet>", problem_dimension},
    {"apply_boundary_conditions<dirichlet_momentum>", problem_dimension},
    {"apply_boundary_conditions<dirichlet_velocity>", problem_dimension},
    {"apply_boundary_conditions<slip>", problem_dimension},
    {"apply_boundary_conditions<no_slip>", problem_dimension},
    {"apply_boundary_conditions<dynamic>", problem_dimension},
};

constexpr unsigned int n_results = []() {
  unsigned int result = 0;
  for (const auto &quantity : quantities)
    result += quantity.size;
  return result;
}();


/*
 * Some helper callables for testing apply_galilei_transform() and
 * apply_boundary_conditions().
 */

struct GalileiTransform {
  template <typename T>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE T operator()(const T &momentum) const
  {
    /* Rotate the momentum vector by 90 degrees: */
    T result;
    result[0] = -momentum[1];
    result[1] = momentum[0];
    return result;
  }
};


struct DirichletData {
  state_type U_bar;

  DEAL_II_HOST_DEVICE_ALWAYS_INLINE state_type operator()() const
  {
    return U_bar;
  }
};


template <typename View>
DEAL_II_HOST_DEVICE dealii::Tensor<1, n_results, double>
compute_quantities(const View &view,
                   const state_type &U,
                   const state_type &U_bar,
                   const dealii::Tensor<1, dim, double> &normal,
                   const dealii::Tensor<1, dim, double> &c_ij,
                   const typename View::PrecomputedVectorView &pv,
                   const typename View::InitialPrecomputedVectorView &ipv,
                   const unsigned int i)
{
  dealii::Tensor<1, n_results, double> result;
  unsigned int k = 0;

  result[k++] = view.density(U);
  result[k++] = view.filter_vacuum_density(view.density(U));

  {
    const auto momentum = view.momentum(U);
    for (unsigned int d = 0; d < dim; ++d)
      result[k++] = momentum[d];
  }

  result[k++] = view.total_energy(U);
  result[k++] = view.internal_energy(U);

  {
    const auto derivative = view.internal_energy_derivative(U);
    for (unsigned int d = 0; d < problem_dimension; ++d)
      result[k++] = derivative[d];
  }

  result[k++] = view.pressure(U);
  result[k++] = view.speed_of_sound(U);
  result[k++] = view.specific_entropy(U);
  result[k++] = view.harten_entropy(U);

  {
    const auto derivative = view.harten_entropy_derivative(U);
    for (unsigned int d = 0; d < problem_dimension; ++d)
      result[k++] = derivative[d];
  }

  result[k++] = view.mathematical_entropy(U);

  {
    const auto derivative = view.mathematical_entropy_derivative(U);
    for (unsigned int d = 0; d < problem_dimension; ++d)
      result[k++] = derivative[d];
  }

  result[k++] = view.is_admissible(U) ? 1. : 0.;

  {
    const auto f = view.f(U);
    for (unsigned int d = 0; d < problem_dimension; ++d)
      for (unsigned int e = 0; e < dim; ++e)
        result[k++] = f[d][e];
  }

  {
    /* Exercise both flux_contribution() variants: */
    const auto flux_i = view.flux_contribution(pv, ipv, i, U);
    const auto flux_j = view.flux_contribution(pv, ipv, &i, U_bar);
    const auto flux_divergence = view.flux_divergence(flux_i, flux_j, c_ij);
    for (unsigned int d = 0; d < problem_dimension; ++d)
      result[k++] = flux_divergence[d];
  }

  const auto primitive_state = view.to_primitive_state(U);
  for (unsigned int d = 0; d < problem_dimension; ++d)
    result[k++] = primitive_state[d];

  {
    const auto state = view.from_primitive_state(primitive_state);
    for (unsigned int d = 0; d < problem_dimension; ++d)
      result[k++] = state[d];
  }

  {
    /* This also exercises expand_state(): */
    dealii::Tensor<1, 3, double> initial_state;
    initial_state[0] = primitive_state[0];
    initial_state[1] = primitive_state[1];
    initial_state[2] = primitive_state[dim + 1];
    const auto state = view.from_initial_state(initial_state);
    for (unsigned int d = 0; d < problem_dimension; ++d)
      result[k++] = state[d];
  }

  {
    const auto state = view.apply_galilei_transform(U, GalileiTransform{});
    for (unsigned int d = 0; d < problem_dimension; ++d)
      result[k++] = state[d];
  }

  {
    const auto eigenvectors =
        view.template linearized_eigenvector<1>(U, normal);
    for (unsigned int j = 0; j < 2; ++j)
      for (unsigned int d = 0; d < problem_dimension; ++d)
        result[k++] = eigenvectors[j][d];
  }

  {
    const auto eigenvectors =
        view.template linearized_eigenvector<problem_dimension>(U, normal);
    for (unsigned int j = 0; j < 2; ++j)
      for (unsigned int d = 0; d < problem_dimension; ++d)
        result[k++] = eigenvectors[j][d];
  }

  {
    const auto state =
        view.template prescribe_riemann_characteristic<1>(U, U_bar, normal);
    for (unsigned int d = 0; d < problem_dimension; ++d)
      result[k++] = state[d];
  }

  {
    const auto state =
        view.template prescribe_riemann_characteristic<2>(U_bar, U, normal);
    for (unsigned int d = 0; d < problem_dimension; ++d)
      result[k++] = state[d];
  }

  {
    /*
     * Only iterate over boundary ids that are actually implemented in
     * apply_boundary_conditions().
     */
    constexpr dealii::types::boundary_id ids[]{Boundary::dirichlet,
                                               Boundary::dirichlet_momentum,
                                               Boundary::dirichlet_velocity,
                                               Boundary::slip,
                                               Boundary::no_slip,
                                               Boundary::dynamic};

    const DirichletData dirichlet_data{U_bar};
    for (const auto id : ids) {
      const auto state =
          view.apply_boundary_conditions(id, U, normal, dirichlet_data);
      for (unsigned int d = 0; d < problem_dimension; ++d)
        result[k++] = state[d];
    }
  }

  Assert(k == n_results, dealii::ExcInternalError());
  return result;
}


int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  std::cout << std::setprecision(10);
  std::cout << std::scientific;

  Euler::HyperbolicSystem hyperbolic_system;

  const auto host_view = hyperbolic_system.view<dim, double, HostSpace>();
  const auto device_view = hyperbolic_system.view<dim, double, DefaultSpace>();

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

  typename HostView::PrecomputedVector precomputed;
  precomputed.reinit_with_scalar_partitioner(scalar_partitioner);

  /* Note: the Euler equations have no precomputed initial values. */
  typename HostView::InitialPrecomputedVector initial_precomputed;
  initial_precomputed.reinit_with_scalar_partitioner(scalar_partitioner);

  Vectors::MultiComponentVector<double, n_constants> constants;
  constants.reinit_with_scalar_partitioner(scalar_partitioner);

  Vectors::MultiComponentVector<double, n_results> results;
  results.reinit_with_scalar_partitioner(scalar_partitioner);

  /* Fill states and precomputed values on the host space: */
  {
    const auto U_view = U.view<HostSpace>();
    const auto precomputed_view = precomputed.view<HostSpace>();

    for (unsigned int i = 0; i < n_states; ++i) {
      state_type primitive;
      primitive[0] = 1. + 0.125 * i;
      primitive[1] = 0.1 * i;
      primitive[2] = -0.05 * i;
      primitive[3] = 1. + 0.25 * i;
      const auto U_i = host_view.from_primitive_state(primitive);
      U_view.write_tensor(U_i, i);

      typename HostView::precomputed_type prec_i;
      prec_i[0] = host_view.specific_entropy(U_i);
      prec_i[1] = host_view.harten_entropy(U_i);
      precomputed_view.write_tensor(prec_i, i);
    }
  }

  /* A second state, a normal, and a c_ij used for the computations below: */

  state_type U_bar;
  {
    state_type primitive;
    primitive[0] = 1.4;
    primitive[1] = 0.3;
    primitive[2] = -0.2;
    primitive[3] = 1.0;
    U_bar = host_view.from_primitive_state(primitive);
  }

  dealii::Tensor<1, dim, double> normal;
  normal[0] = 0.6;
  normal[1] = -0.8;

  dealii::Tensor<1, dim, double> c_ij;
  c_ij[0] = 0.25;
  c_ij[1] = 0.5;

  /* Compute all quantities on the host: */

  const auto host_constants = compute_constants(host_view);

  std::array<dealii::Tensor<1, n_results, double>, n_states> host_results;
  {
    const auto pv = precomputed.view<HostSpace>();
    const auto ipv = initial_precomputed.view<HostSpace>();

    const auto U_view = U.view<HostSpace>();
    for (unsigned int i = 0; i < n_states; ++i) {
      const auto U_i = U_view.read_tensor<double>(i);
      host_results[i] =
          compute_quantities(host_view, U_i, U_bar, normal, c_ij, pv, ipv, i);
    }
  }

  /* Compute the same quantities on the default space: */

  U.move_to_memory_space<DefaultSpace>();
  precomputed.move_to_memory_space<DefaultSpace>();
  initial_precomputed.move_to_memory_space<DefaultSpace>();
  constants.move_to_memory_space<DefaultSpace>();
  results.move_to_memory_space<DefaultSpace>();

  {
    const auto U_view = U.view<DefaultSpace>();
    const auto pv = precomputed.view<DefaultSpace>();
    const auto ipv = initial_precomputed.view<DefaultSpace>();
    const auto constants_view = constants.view<DefaultSpace>();
    const auto results_view = results.view<DefaultSpace>();

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
          const auto U_i = U_view.read_tensor(i);
          const auto result = compute_quantities(
              device_view, U_i, U_bar, normal, c_ij, pv, ipv, i);
          results_view.write_tensor(result, i);
        });
  }

  constants.move_to_memory_space<HostSpace>();
  results.move_to_memory_space<HostSpace>();

  const auto constants_view = constants.view<HostSpace>();
  const auto results_view = results.view<HostSpace>();

  const unsigned int index = 0;
  const auto device_constants = constants_view.read_tensor<double>(index);

  std::array<dealii::Tensor<1, n_results, double>, n_states> device_results;
  for (unsigned int i = 0; i < n_states; ++i)
    device_results[i] = results_view.read_tensor<double>(i);

  /* Print all results: */

  std::cout << "Runtime parameters and cached inverses:\n\n";
  for (unsigned int k = 0; k < n_constants; ++k) {
    std::cout << constant_names[k] << " (host):   " << host_constants[k]
              << "\n";
    std::cout << constant_names[k] << " (device): " << device_constants[k]
              << "\n";
  }

  std::cout << "\nDerived quantities for " << n_states << " states:\n";
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
