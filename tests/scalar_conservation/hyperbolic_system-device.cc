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
// We select a nondefault flux with a nondefault derivative approximation
// delta so that the parse_parameters_call_back() update path is exercised.
// The flux and its gradient are evaluated on the host and stored as
// precomputed values.
//

using namespace ryujin;

using HostSpace = dealii::MemorySpace::Host;
using DefaultSpace = dealii::MemorySpace::Default;

constexpr int dim = 2;
constexpr unsigned int problem_dimension = 1;
constexpr unsigned int n_states = 8;

using HostView =
    ScalarConservation::HyperbolicSystemView<dim, double, HostSpace>;
using DeviceView =
    ScalarConservation::HyperbolicSystemView<dim, double, DefaultSpace>;
using state_type = typename HostView::state_type;
using precomputed_type = typename HostView::precomputed_type;


/*
 * Runtime parameters. These do not depend on a state and are thus
 * computed separately:
 */

constexpr const char *constant_names[]{"derivative_approximation_delta"};

constexpr unsigned int n_constants = std::size(constant_names);


template <typename View>
DEAL_II_HOST_DEVICE dealii::Tensor<1, n_constants, double>
compute_constants(const View &view)
{
  dealii::Tensor<1, n_constants, double> result;
  unsigned int k = 0;

  result[k++] = view.derivative_approximation_delta();

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
    {"state", 1},
    {"square_entropy", 1},
    {"square_entropy_derivative", 1},
    {"kruzkov_entropy", 1},
    {"kruzkov_entropy_derivative", 1},
    {"is_admissible", 1},
    {"construct_flux_tensor", dim},
    {"construct_flux_gradient_tensor", dim},
    {"flux_contribution (i)", problem_dimension *dim},
    {"flux_contribution (js)", problem_dimension *dim},
    {"flux_divergence", problem_dimension},
    {"expand_state", problem_dimension},
    {"to_primitive_state", problem_dimension},
    {"from_primitive_state", problem_dimension},
    {"apply_galilei_transform", problem_dimension},
    {"apply_boundary_conditions<dirichlet>", problem_dimension},
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
    return momentum;
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
                   const unsigned int i,
                   const unsigned int j)
{
  dealii::Tensor<1, n_results, double> result;
  unsigned int k = 0;

  const auto u = view.state(U);
  const auto u_bar = view.state(U_bar);

  result[k++] = u;
  result[k++] = view.square_entropy(u);
  result[k++] = view.square_entropy_derivative(u);
  result[k++] = view.kruzkov_entropy(u_bar, u);
  result[k++] = view.kruzkov_entropy_derivative(u_bar, u);
  result[k++] = view.is_admissible(U) ? 1. : 0.;

  {
    const auto prec_i = pv.template read_tensor<double, precomputed_type>(i);

    const auto f = view.construct_flux_tensor(prec_i);
    for (unsigned int d = 0; d < dim; ++d)
      result[k++] = f[d];

    const auto df = view.construct_flux_gradient_tensor(prec_i);
    for (unsigned int d = 0; d < dim; ++d)
      result[k++] = df[d];
  }

  /* Exercise both flux_contribution() variants: */
  const auto flux_i = view.flux_contribution(pv, ipv, i, U);
  const auto flux_j = view.flux_contribution(pv, ipv, &j, U_bar);

  for (unsigned int d = 0; d < problem_dimension; ++d)
    for (unsigned int e = 0; e < dim; ++e)
      result[k++] = flux_i[d][e];

  for (unsigned int d = 0; d < problem_dimension; ++d)
    for (unsigned int e = 0; e < dim; ++e)
      result[k++] = flux_j[d][e];

  {
    const auto flux_divergence = view.flux_divergence(flux_i, flux_j, c_ij);
    for (unsigned int d = 0; d < problem_dimension; ++d)
      result[k++] = flux_divergence[d];
  }

  {
    const auto state = view.expand_state(U);
    for (unsigned int d = 0; d < problem_dimension; ++d)
      result[k++] = state[d];
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
    const auto state = view.apply_galilei_transform(U, GalileiTransform{});
    for (unsigned int d = 0; d < problem_dimension; ++d)
      result[k++] = state[d];
  }

  {
    /*
     * Only Dirichlet boundary conditions are implemented in
     * apply_boundary_conditions().
     */
    const DirichletData dirichlet_data{U_bar};
    const auto state = view.apply_boundary_conditions(
        Boundary::dirichlet, U, normal, dirichlet_data);
    for (unsigned int d = 0; d < problem_dimension; ++d)
      result[k++] = state[d];
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

  /* Exercise the parse_parameters_call_back() update path: */

  std::stringstream parameters;
  parameters << "subsection HyperbolicSystem\n"
             << "set flux = function\n"
             << "subsection function\n"
             << "set expression = 0.5*u*u;u*u*u/3\n"
             << "set derivative approximation delta = 1.0e-4\n"
             << "end\n"
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

  /* We do not have any precomputed initial values: */
  typename HostView::InitialPrecomputedVector initial_precomputed;
  initial_precomputed.reinit_with_scalar_partitioner(scalar_partitioner);

  Vectors::MultiComponentVector<double, n_constants> constants;
  constants.reinit_with_scalar_partitioner(scalar_partitioner);

  Vectors::MultiComponentVector<double, n_results> results;
  results.reinit_with_scalar_partitioner(scalar_partitioner);

  /*
   * Fill states and precomputed values on the host space. The flux and
   * its gradient are computed by calling into the selected flux, which is
   * only possible on the host:
   */
  {
    const auto U_view = U.view<HostSpace>();
    const auto precomputed_view = precomputed.view<HostSpace>();

    for (unsigned int i = 0; i < n_states; ++i) {
      state_type U_i;
      U_i[0] = -1.5 + 0.5 * i;
      U_view.write_tensor(U_i, i);

      const auto u_i = host_view.state(U_i);
      const auto f_i = host_view.flux_function(u_i);
      const auto df_i = host_view.flux_gradient_function(u_i);

      precomputed_type prec_i;
      for (unsigned int d = 0; d < dim; ++d) {
        prec_i[d] = f_i[d];
        prec_i[dim + d] = df_i[d];
      }
      precomputed_view.write_tensor(prec_i, i);
    }
  }

  /*
   * A second state, a normal, and a c_ij used for the computations below:
   */

  state_type U_bar;
  U_bar[0] = 0.7;

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
      const unsigned int j = (i + 1) % n_states;
      const auto U_i = U_view.read_tensor<double>(i);
      host_results[i] = compute_quantities(
          host_view, U_i, U_bar, normal, c_ij, pv, ipv, i, j);
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
              device_view, U_i, U_bar, normal, c_ij, pv, ipv, i, j);
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
