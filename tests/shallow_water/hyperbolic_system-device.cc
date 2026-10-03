#include <hyperbolic_system.h>
#include <multicomponent_vector.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <sstream>

//
// Test that the HyperbolicSystemView can be used on the device memory
// space: We compute all runtime parameters and derived quantities of the
// view on the host and on the default memory space and print both results.
//
// We set nondefault runtime parameters so that the
// parse_parameters_call_back() update path is exercised. The first state
// is a dry state.
//

using namespace ryujin;

using HostSpace = dealii::MemorySpace::Host;
using DefaultSpace = dealii::MemorySpace::Default;

constexpr int dim = 2;
constexpr unsigned int problem_dimension = 1 + dim;
constexpr unsigned int n_states = 8;

using HostView = ShallowWater::HyperbolicSystemView<dim, double, HostSpace>;
using DeviceView =
    ShallowWater::HyperbolicSystemView<dim, double, DefaultSpace>;
using state_type = typename HostView::state_type;
using precomputed_type = typename HostView::precomputed_type;
using initial_precomputed_type = typename HostView::initial_precomputed_type;


/*
 * Runtime parameters. These do not depend on a state and are thus
 * computed separately:
 */

constexpr const char *constant_names[]{"gravity",
                                       "manning_friction_coefficient",
                                       "reference_water_depth",
                                       "dry_state_relaxation_small",
                                       "dry_state_relaxation_large"};

constexpr unsigned int n_constants = std::size(constant_names);


template <typename View>
DEAL_II_HOST_DEVICE dealii::Tensor<1, n_constants, double>
compute_constants(const View &view)
{
  dealii::Tensor<1, n_constants, double> result;
  unsigned int k = 0;

  result[k++] = view.gravity();
  result[k++] = view.manning_friction_coefficient();
  result[k++] = view.reference_water_depth();
  result[k++] = view.dry_state_relaxation_small();
  result[k++] = view.dry_state_relaxation_large();

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
    {"water_depth", 1},
    {"inverse_water_depth_mollified", 1},
    {"water_depth_sharp", 1},
    {"inverse_water_depth_sharp", 1},
    {"filter_dry_water_depth", 1},
    {"momentum", dim},
    {"kinetic_energy", 1},
    {"pressure", 1},
    {"speed_of_sound", 1},
    {"mathematical_entropy", 1},
    {"mathematical_entropy_derivative", problem_dimension},
    {"is_admissible", 1},
    {"f", problem_dimension *dim},
    {"g", problem_dimension *dim},
    {"star_state", problem_dimension},
    {"equilibrated_states", 2 * problem_dimension},
    {"flux_divergence", problem_dimension},
    {"high_order_flux_divergence", problem_dimension},
    {"affine_shift", problem_dimension},
    {"manning_friction", problem_dimension},
    {"nodal_source (i)", problem_dimension},
    {"nodal_source (js)", problem_dimension},
    {"to_primitive_state", problem_dimension},
    {"from_primitive_state", problem_dimension},
    {"from_initial_state", problem_dimension},
    {"apply_galilei_transform", problem_dimension},
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
                   const double d_ij,
                   const double tau,
                   const typename View::PrecomputedVectorView &pv,
                   const typename View::InitialPrecomputedVectorView &ipv,
                   const unsigned int i,
                   const unsigned int j)
{
  dealii::Tensor<1, n_results, double> result;
  unsigned int k = 0;

  const auto &[eta_m, h_star] =
      pv.template read_tensor<double, precomputed_type>(i);

  result[k++] = view.water_depth(U);
  result[k++] = view.inverse_water_depth_mollified(U);
  result[k++] = view.water_depth_sharp(U);
  result[k++] = view.inverse_water_depth_sharp(U);
  result[k++] = view.filter_dry_water_depth(view.water_depth(U));

  {
    const auto momentum = view.momentum(U);
    for (unsigned int d = 0; d < dim; ++d)
      result[k++] = momentum[d];
  }

  result[k++] = view.kinetic_energy(U);
  result[k++] = view.pressure(U);
  result[k++] = view.speed_of_sound(U);
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
    const auto g = view.g(U);
    for (unsigned int d = 0; d < problem_dimension; ++d)
      for (unsigned int e = 0; e < dim; ++e)
        result[k++] = g[d][e];
  }

  /*
   * Exercise both flux_contribution() variants. The flux contribution
   * consists of the state and the bathymetry:
   */
  const auto flux_i = view.flux_contribution(pv, ipv, i, U);
  const auto flux_j = view.flux_contribution(pv, ipv, &j, U_bar);

  {
    const auto &[U_i, Z_i] = flux_i;
    const auto &[U_j, Z_j] = flux_j;
    const auto state = view.star_state(U_i, Z_i, Z_j);
    for (unsigned int d = 0; d < problem_dimension; ++d)
      result[k++] = state[d];
  }

  {
    const auto states = view.equilibrated_states(flux_i, flux_j);
    for (unsigned int l = 0; l < 2; ++l)
      for (unsigned int d = 0; d < problem_dimension; ++d)
        result[k++] = states[l][d];
  }

  {
    const auto flux_divergence = view.flux_divergence(flux_i, flux_j, c_ij);
    for (unsigned int d = 0; d < problem_dimension; ++d)
      result[k++] = flux_divergence[d];
  }

  {
    const auto flux_divergence =
        view.high_order_flux_divergence(flux_i, flux_j, c_ij);
    for (unsigned int d = 0; d < problem_dimension; ++d)
      result[k++] = flux_divergence[d];
  }

  {
    const auto affine_shift = view.affine_shift(flux_i, flux_j, c_ij, d_ij);
    for (unsigned int d = 0; d < problem_dimension; ++d)
      result[k++] = affine_shift[d];
  }

  {
    const auto source = view.manning_friction(U, h_star, tau);
    for (unsigned int d = 0; d < problem_dimension; ++d)
      result[k++] = source[d];
  }

  {
    /* Exercise both nodal_source() variants: */
    const auto source_i = view.nodal_source(pv, i, U, tau);
    for (unsigned int d = 0; d < problem_dimension; ++d)
      result[k++] = source_i[d];

    const auto source_j = view.nodal_source(pv, &j, U_bar, tau);
    for (unsigned int d = 0; d < problem_dimension; ++d)
      result[k++] = source_j[d];
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
    /* Exercise expand_state() with a one dimensional initial state: */
    dealii::Tensor<1, 2, double> initial_state;
    initial_state[0] = primitive_state[0];
    initial_state[1] = primitive_state[1];
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

  ShallowWater::HyperbolicSystem hyperbolic_system;

  /* Exercise the parse_parameters_call_back() update path: */

  std::stringstream parameters;
  parameters << "subsection HyperbolicSystem\n"
             << "set gravity = 10.0\n"
             << "set manning friction coefficient = 0.05\n"
             << "set reference water depth = 2.0\n"
             << "set dry state relaxation small = 50\n"
             << "set dry state relaxation large = 5000\n"
             << "end" << std::endl;
  dealii::ParameterAcceptor::initialize(parameters);

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

  /* The precomputed initial values store the bathymetry: */
  typename HostView::InitialPrecomputedVector initial_precomputed;
  initial_precomputed.reinit_with_scalar_partitioner(scalar_partitioner);

  Vectors::MultiComponentVector<double, n_constants> constants;
  constants.reinit_with_scalar_partitioner(scalar_partitioner);

  Vectors::MultiComponentVector<double, n_results> results;
  results.reinit_with_scalar_partitioner(scalar_partitioner);

  /*
   * Fill states, precomputed values, and the bathymetry on the host
   * space. The first state is a dry state. The bathymetry is chosen such
   * that it increases from every wet state i to its neighbor (i + 1) %
   * n_states:
   */
  {
    const auto U_view = U.view<HostSpace>();
    const auto precomputed_view = precomputed.view<HostSpace>();
    const auto initial_precomputed_view = initial_precomputed.view<HostSpace>();

    for (unsigned int i = 0; i < n_states; ++i) {
      state_type primitive;
      primitive[0] = 0.25 * i;
      primitive[1] = 0.8 * i;
      primitive[2] = -0.4 * i;
      const auto U_i = host_view.from_primitive_state(primitive);
      U_view.write_tensor(U_i, i);

      const precomputed_type prec_i{
          host_view.mathematical_entropy(U_i),
          ryujin::pow(host_view.water_depth_sharp(U_i), 4. / 3.)};
      precomputed_view.write_tensor(prec_i, i);

      const initial_precomputed_type bathymetry_i{0.05 * ((i + 7) % n_states)};
      initial_precomputed_view.write_tensor(bathymetry_i, i);
    }
  }

  /*
   * A second state, a normal, a c_ij, a d_ij, and a time-step size used
   * for the computations below:
   */

  state_type U_bar;
  {
    state_type primitive;
    primitive[0] = 1.4;
    primitive[1] = 0.3;
    primitive[2] = -0.2;
    U_bar = host_view.from_primitive_state(primitive);
  }

  dealii::Tensor<1, dim, double> normal;
  normal[0] = 0.6;
  normal[1] = -0.8;

  dealii::Tensor<1, dim, double> c_ij;
  c_ij[0] = 0.25;
  c_ij[1] = 0.5;

  const double d_ij = 0.75;
  const double tau = 0.1;

  /* Compute all quantities on the host: */

  const auto host_constants = compute_constants(host_view);

  std::array<dealii::Tensor<1, n_results, double>, n_states> host_results;
  {
    const auto pv = precomputed.view<HostSpace>();
    const auto ipv = initial_precomputed.view<HostSpace>();

    const auto U_view = U.view<HostSpace>();
    for (unsigned int i = 0; i < n_states; ++i) {
      const unsigned int j = (i + 1) % n_states;
      const auto U_i = U_view.read_tensor<double>(i);
      host_results[i] = compute_quantities(
          host_view, U_i, U_bar, normal, c_ij, d_ij, tau, pv, ipv, i, j);
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
          const unsigned int j = (i + 1) % n_states;
          const auto U_i = U_view.read_tensor(i);
          const auto result = compute_quantities(
              device_view, U_i, U_bar, normal, c_ij, d_ij, tau, pv, ipv, i, j);
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

  std::cout << "Runtime parameters:\n\n";
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
