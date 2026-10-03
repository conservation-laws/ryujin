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
// results. We set a nonzero number of Newton iterations so that the test
// exercises the full quadratic Newton solver.
//
// We select a van der Waals equation of state with a nonzero covolume so
// that the surrogate gamma varies from state to state. We run the test with
// strict and with non-strict bounds, which only differ for varying gamma.
//

using namespace ryujin;

using HostSpace = dealii::MemorySpace::Host;
using DefaultSpace = dealii::MemorySpace::Default;

constexpr int dim = 2;
constexpr unsigned int problem_dimension = 2 + dim;
constexpr unsigned int n_states = 8;

using HostView = EulerAEOS::WaveSpeedEstimatorView<dim, double, HostSpace>;
using DeviceView = EulerAEOS::WaveSpeedEstimatorView<dim, double, DefaultSpace>;
using HostSystemView = EulerAEOS::HyperbolicSystemView<dim, double, HostSpace>;
using state_type = typename HostSystemView::state_type;


/*
 * Runtime parameters of the Riemann solver. These do not depend on a state
 * and are thus computed separately:
 */

constexpr const char *constant_names[]{"newton_tolerance",
                                       "newton_max_iterations"};

constexpr unsigned int n_constants = std::size(constant_names);


template <typename View>
DEAL_II_HOST_DEVICE dealii::Tensor<1, n_constants, double>
compute_constants(const View &view)
{
  dealii::Tensor<1, n_constants, double> result;
  unsigned int k = 0;

  result[k++] = view.newton_tolerance();
  result[k++] = view.newton_max_iterations();

  Assert(k == n_constants, dealii::ExcInternalError());
  return result;
}


constexpr unsigned int n_results = 1; /* lambda_max */


template <typename View>
DEAL_II_HOST_DEVICE dealii::Tensor<1, n_results, double>
compute_quantities(const View &wave_speed_estimator_view,
                   const typename View::PrecomputedVectorView &pv,
                   const unsigned int i,
                   const state_type &U_i,
                   const unsigned int j,
                   const state_type &U_j,
                   const dealii::Tensor<1, dim, double> &n_ij)
{
  dealii::Tensor<1, n_results, double> result;
  unsigned int k = 0;

  result[k++] = wave_speed_estimator_view.compute(pv, U_i, U_j, i, &j, n_ij);

  Assert(k == n_results, dealii::ExcInternalError());
  return result;
}


void run(const EulerAEOS::HyperbolicSystem &hyperbolic_system,
         const EulerAEOS::WaveSpeedEstimator<double> &wave_speed_estimator,
         const EulerAEOS::NASGRiemannSolver<double> &riemann_solver)
{
  const auto host_system_view =
      hyperbolic_system.view<dim, double, HostSpace>();
  const auto host_view = wave_speed_estimator.view<dim, double, HostSpace>();
  const auto device_view =
      wave_speed_estimator.view<dim, double, DefaultSpace>();

  const auto host_riemann_solver_view =
      riemann_solver.view<double, HostSpace>();
  const auto device_riemann_solver_view =
      riemann_solver.view<double, DefaultSpace>();

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

  Vectors::MultiComponentVector<double, n_constants> constants;
  constants.reinit_with_scalar_partitioner(scalar_partitioner);

  Vectors::MultiComponentVector<double, n_results> results;
  results.reinit_with_scalar_partitioner(scalar_partitioner);

  /*
   * Fill states and precomputed values on the host space. The pressure
   * has to be computed with the equation of state oracle on the host:
   */
  {
    const auto U_view = U.view();
    const auto precomputed_view = precomputed.view();

    for (unsigned int i = 0; i < n_states; ++i) {
      state_type primitive;
      primitive[0] = 1. + 0.125 * i;
      primitive[1] = 0.1 * i;
      primitive[2] = -0.05 * i;
      primitive[3] = 1. + 0.25 * i;
      const auto U_i = host_system_view.from_primitive_state(primitive);
      U_view.write_tensor(U_i, i);

      const auto rho_i = host_system_view.density(U_i);
      const auto e_i = host_system_view.internal_energy(U_i) / rho_i;
      const auto p_i = host_system_view.eos_pressure(rho_i, e_i);
      const auto gamma_i = host_system_view.surrogate_gamma(U_i, p_i);

      const typename HostSystemView::precomputed_type prec_i{
          p_i,
          gamma_i,
          host_system_view.surrogate_specific_entropy(U_i, gamma_i),
          host_system_view.surrogate_harten_entropy(U_i, gamma_i)};
      precomputed_view.write_tensor(prec_i, i);
    }
  }

  /* A normal used for the computations below: */

  dealii::Tensor<1, dim, double> normal;
  normal[0] = 0.6;
  normal[1] = -0.8;

  /* Compute all quantities on the host: */

  const auto host_constants = compute_constants(host_riemann_solver_view);

  std::array<dealii::Tensor<1, n_results, double>, n_states> host_results;
  {
    const auto U_view = U.view<HostSpace>();
    const auto pv = precomputed.view<HostSpace>();

    for (unsigned int i = 0; i < n_states; ++i) {
      const unsigned int j = (i + 3) % n_states;
      const auto U_i = U_view.read_tensor<double>(i);
      const auto U_j = U_view.read_tensor<double>(j);
      host_results[i] =
          compute_quantities(host_view, pv, i, U_i, j, U_j, normal);
    }
  }

  /* Compute the same quantities on the default space: */

  U.move_to_memory_space<DefaultSpace>();
  precomputed.move_to_memory_space<DefaultSpace>();
  constants.move_to_memory_space<DefaultSpace>();
  results.move_to_memory_space<DefaultSpace>();

  const auto U_view = U.view<DefaultSpace>();
  const auto pv = precomputed.view<DefaultSpace>();
  const auto constants_view = constants.view<DefaultSpace>();
  const auto results_view = results.view<DefaultSpace>();

  using ExecutionSpace = DefaultSpace::kokkos_space::execution_space;
  const auto exec = ExecutionSpace{};

  Kokkos::parallel_for(
      "test_constants",
      Kokkos::RangePolicy<ExecutionSpace>(exec, 0, 1),
      KOKKOS_LAMBDA(std::size_t i) {
        constants_view.write_tensor(
            compute_constants(device_riemann_solver_view), i);
      });

  Kokkos::parallel_for(
      "test_quantities",
      Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n_states),
      KOKKOS_LAMBDA(std::size_t i) {
        const unsigned int j = (i + 3) % n_states;
        const auto U_i = U_view.read_tensor(i);
        const auto U_j = U_view.read_tensor(j);
        const auto result =
            compute_quantities(device_view, pv, i, U_i, j, U_j, normal);
        results_view.write_tensor(result, i);
      });

  constants.move_to_memory_space<HostSpace>();
  results.move_to_memory_space<HostSpace>();

  const unsigned int index = 0;
  const auto device_constants =
      constants.view<HostSpace>().read_tensor<double>(index);

  std::array<dealii::Tensor<1, n_results, double>, n_states> device_results;
  {
    const auto results_host_view = results.view<HostSpace>();
    for (unsigned int i = 0; i < n_states; ++i)
      device_results[i] = results_host_view.read_tensor<double>(i);
  }

  /* Print all results: */

  std::cout << "Runtime parameters:\n\n";
  for (unsigned int k = 0; k < n_constants; ++k) {
    std::cout << constant_names[k] << " (host):   " << host_constants[k]
              << "\n";
    std::cout << constant_names[k] << " (device): " << device_constants[k]
              << "\n";
  }

  std::cout << "\nlambda_max for " << n_states << " states:\n";
  std::cout << "\nlambda_max (host):  ";
  for (unsigned int i = 0; i < n_states; ++i)
    std::cout << " " << host_results[i][0];
  std::cout << "\nlambda_max (device):";
  for (unsigned int i = 0; i < n_states; ++i)
    std::cout << " " << device_results[i][0];
  std::cout << "\n";
}


int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  std::cout << std::setprecision(10);
  std::cout << std::scientific;

  EulerAEOS::HyperbolicSystem hyperbolic_system;
  EulerAEOS::WaveSpeedEstimator<double> wave_speed_estimator(hyperbolic_system);
  EulerAEOS::NASGRiemannSolver<double> riemann_solver("/RiemannSolver");

  /*
   * Exercise the parse_parameters_call_back() update path and enable the
   * quadratic Newton solver:
   */

  for (const bool strict : {true, false}) {
    std::stringstream parameters;
    parameters << "subsection HyperbolicSystem\n"
               << "set equation of state = van der waals\n"
               << "set compute strict bounds = " << (strict ? "true" : "false")
               << "\n"
               << "subsection van der waals\n"
               << "set covolume b = 0.1\n"
               << "set vdw a = 0.1\n"
               << "end\n"
               << "end\n"
               << "subsection WaveSpeedEstimator\n"
               << "set newton max iterations = 2\n"
               << "end\n"
               << "subsection RiemannSolver\n"
               << "set newton max iterations = 2\n"
               << "end" << std::endl;
    dealii::ParameterAcceptor::initialize(parameters);

    std::cout << "\ncompute strict bounds = " << (strict ? "true" : "false")
              << "\n\n";
    run(hyperbolic_system, wave_speed_estimator, riemann_solver);
  }
}
