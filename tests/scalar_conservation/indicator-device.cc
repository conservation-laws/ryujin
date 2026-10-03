#include <indicator.h>
#include <multicomponent_vector.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <sstream>

//
// Test that the IndicatorView can be used on the device memory space: We
// run the stencil-based indicator (reset(), accumulate(), alpha()) over a
// set of states on the host and on the default memory space and print both
// results.
//
// We set nondefault runtime parameters. The flux and its gradient are
// evaluated on the host and stored as precomputed values.
//

using namespace ryujin;

using HostSpace = dealii::MemorySpace::Host;
using DefaultSpace = dealii::MemorySpace::Default;

constexpr int dim = 2;
constexpr unsigned int problem_dimension = 1;
constexpr unsigned int n_states = 8;

using HostView = ScalarConservation::IndicatorView<dim, double, HostSpace>;
using DeviceView = ScalarConservation::IndicatorView<dim, double, DefaultSpace>;
using HostSystemView =
    ScalarConservation::HyperbolicSystemView<dim, double, HostSpace>;
using state_type = typename HostSystemView::state_type;

constexpr const char *quantity_names[]{"evc_factor", "alpha"};
constexpr unsigned int n_results = std::size(quantity_names);


template <typename View>
DEAL_II_HOST_DEVICE dealii::Tensor<1, n_results, double>
compute_quantities(const View &indicator_view,
                   const typename View::PrecomputedVectorView &pv,
                   const unsigned int i,
                   const state_type &U_i,
                   const unsigned int j_1,
                   const state_type &U_j_1,
                   const unsigned int j_2,
                   const state_type &U_j_2,
                   const dealii::Tensor<1, dim, double> &c_ij_1,
                   const dealii::Tensor<1, dim, double> &c_ij_2,
                   const double hd_i)
{
  dealii::Tensor<1, n_results, double> result;
  unsigned int k = 0;

  /* The view is stateful, work on a copy: */
  auto indicator = indicator_view;

  indicator.reset(pv, i, U_i);
  indicator.accumulate(pv, &j_1, U_j_1, c_ij_1);
  indicator.accumulate(pv, &j_2, U_j_2, c_ij_2);

  result[k++] = indicator.evc_factor();
  result[k++] = indicator.alpha(hd_i);

  Assert(k == n_results, dealii::ExcInternalError());
  return result;
}


int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  std::cout << std::setprecision(10);
  std::cout << std::scientific;

  ScalarConservation::HyperbolicSystem hyperbolic_system;
  ScalarConservation::Indicator<double> indicator(hyperbolic_system);

  /* Exercise the parse_parameters_call_back() update path: */

  std::stringstream parameters;
  parameters << "subsection HyperbolicSystem\n"
             << "set flux = function\n"
             << "subsection function\n"
             << "set expression = 0.5*u*u;u*u*u/3\n"
             << "set derivative approximation delta = 1.0e-4\n"
             << "end\n"
             << "end\n"
             << "subsection Indicator\n"
             << "set evc factor = 0.75\n"
             << "end" << std::endl;
  dealii::ParameterAcceptor::initialize(parameters);

  const auto host_system_view =
      hyperbolic_system.view<dim, double, HostSpace>();
  const auto host_view = indicator.view<dim, double, HostSpace>();
  const auto device_view = indicator.view<dim, double, DefaultSpace>();

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

  /* Two c_ij vectors and a scaled measure used for the stencil: */

  dealii::Tensor<1, dim, double> c_ij_1;
  c_ij_1[0] = 0.25;
  c_ij_1[1] = 0.5;

  dealii::Tensor<1, dim, double> c_ij_2;
  c_ij_2[0] = -0.5;
  c_ij_2[1] = 0.125;

  const double hd_i = 0.01;

  /* Compute all quantities on the host: */

  std::array<dealii::Tensor<1, n_results, double>, n_states> host_results;
  {
    const auto U_view = U.view<HostSpace>();
    const auto pv = precomputed.view<HostSpace>();

    for (unsigned int i = 0; i < n_states; ++i) {
      const unsigned int j_1 = (i + 1) % n_states;
      const unsigned int j_2 = (i + 7) % n_states;
      const auto U_i = U_view.read_tensor<double>(i);
      const auto U_j_1 = U_view.read_tensor<double>(j_1);
      const auto U_j_2 = U_view.read_tensor<double>(j_2);
      host_results[i] = compute_quantities(
          host_view, pv, i, U_i, j_1, U_j_1, j_2, U_j_2, c_ij_1, c_ij_2, hd_i);
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
        const unsigned int j_1 = (i + 1) % n_states;
        const unsigned int j_2 = (i + 7) % n_states;
        const auto U_i = U_view.read_tensor(i);
        const auto U_j_1 = U_view.read_tensor(j_1);
        const auto U_j_2 = U_view.read_tensor(j_2);
        const auto result = compute_quantities(device_view,
                                               pv,
                                               i,
                                               U_i,
                                               j_1,
                                               U_j_1,
                                               j_2,
                                               U_j_2,
                                               c_ij_1,
                                               c_ij_2,
                                               hd_i);
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

  std::cout << "Indicator quantities for " << n_states << " states:\n";
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
