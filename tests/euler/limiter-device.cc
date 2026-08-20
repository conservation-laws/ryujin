#include <hyperbolic_system.h>
#include <limiter.h>
#include <multicomponent_vector.h>

#include <iomanip>
#include <iostream>

int main(int argc, char *argv[])
{
  //
  // Test that the LimiterView can be used on the device memory space:
  //

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  using HostSpace = dealii::MemorySpace::Host;
  using DefaultSpace = dealii::MemorySpace::Default;

  constexpr int dim = 2;
  constexpr unsigned int problem_dimension = 2 + dim;
  constexpr unsigned int n_states = 8;

  std::cout << std::setprecision(10);
  std::cout << std::scientific;

  ryujin::Euler::HyperbolicSystem hyperbolic_system;
  ryujin::Euler::Limiter<double> limiter(hyperbolic_system);

  const auto hs_view = hyperbolic_system.view<dim, double, HostSpace>();
  const auto host_view = limiter.view<dim, double, HostSpace>();
  const auto device_view = limiter.view<dim, double, DefaultSpace>();

  using View = ryujin::Euler::HyperbolicSystemView<dim, double>;
  using state_type = typename View::state_type;
  using precomputed_type = typename View::precomputed_type;
  using flux_contribution_type = typename View::flux_contribution_type;
  constexpr auto n_precomputed_values = View::n_precomputed_values;

  /* Set up locally owned and relevant index sets. */

  dealii::IndexSet locally_owned(n_states);
  dealii::IndexSet locally_relevant(n_states);
  locally_owned.add_range(0, n_states);
  locally_relevant.add_range(0, n_states);

  const auto scalar_partitioner =
      std::make_shared<dealii::Utilities::MPI::Partitioner>(
          locally_owned, locally_relevant, MPI_COMM_WORLD);

  ryujin::Vectors::MultiComponentVector<double, problem_dimension> U;
  U.reinit_with_scalar_partitioner(scalar_partitioner);

  ryujin::Vectors::MultiComponentVector<double, n_precomputed_values> pv;
  pv.reinit_with_scalar_partitioner(scalar_partitioner);

  ryujin::Vectors::MultiComponentVector<double, 4> results;
  results.reinit_with_scalar_partitioner(scalar_partitioner);

  /* Fill states and precomputed values on the host space: */

  for (unsigned int i = 0; i < n_states; ++i) {
    state_type primitive;
    primitive[0] = 1. + 0.125 * i;
    primitive[1] = 0.1 * i;
    primitive[2] = -0.05 * i;
    primitive[3] = 1. + 0.25 * i;
    const auto U_i = hs_view.from_primitive_state(primitive);
    U.write_tensor(U_i, i);

    const precomputed_type prec_i{hs_view.specific_entropy(U_i),
                                  hs_view.harten_entropy(U_i)};
    pv.write_tensor(prec_i, i);
  }

  constexpr double hd_i = 0.25;

  /* Accumulate bounds and limit on the host: */

  std::cout << "Computed on the host:\n";
  for (unsigned int i = 0; i < n_states; ++i) {
    auto view = host_view;
    const auto U_i = U.read_tensor<double>(i);
    view.reset(pv, i, U_i, flux_contribution_type{});
    for (unsigned int j = 0; j < n_states; ++j) {
      if (j == i)
        continue;
      unsigned int js[1] = {j};
      const auto U_j = U.read_tensor<double>(js);
      dealii::Tensor<1, dim, double> scaled_c_ij;
      scaled_c_ij[0] = 0.03 * (double(j) - double(i));
      scaled_c_ij[1] = 0.01 * (double(j) + 1.);
      view.accumulate(
          pv, js, U_j, flux_contribution_type{}, scaled_c_ij, state_type{});
    }
    const auto bounds = view.bounds(hd_i);

    unsigned int next[1] = {(i + 1) % n_states};
    const auto P = 4.0 * (U.read_tensor<double>(next) - U_i);
    const auto [l, success] = view.limit(bounds, U_i, P);

    std::cout << bounds[0] << " " << bounds[1] << " " << bounds[2] << " " << l
              << "\n";
  }

  /* Accumulate bounds and limit on the default space: */

  U.move_to_memory_space<DefaultSpace>();
  pv.move_to_memory_space<DefaultSpace>();
  results.move_to_memory_space<DefaultSpace>();

  const auto U_view = U.get_view<DefaultSpace>();
  const auto pv_view = pv.get_view<DefaultSpace>();
  const auto results_view = results.get_view<DefaultSpace>();

  using ExecutionSpace = DefaultSpace::kokkos_space::execution_space;
  const auto exec = ExecutionSpace{};
  Kokkos::parallel_for("test",
                       Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n_states),
                       [=](std::size_t i) {
                         auto view = device_view;
                         const auto U_i = U_view.read_tensor(i);
                         view.reset(pv_view, i, U_i, flux_contribution_type{});
                         for (unsigned int j = 0; j < n_states; ++j) {
                           if (j == i)
                             continue;
                           unsigned int js[1] = {j};
                           const auto U_j = U_view.read_tensor(js);
                           dealii::Tensor<1, dim, double> scaled_c_ij;
                           scaled_c_ij[0] = 0.03 * (double(j) - double(i));
                           scaled_c_ij[1] = 0.01 * (double(j) + 1.);
                           view.accumulate(pv_view,
                                           js,
                                           U_j,
                                           flux_contribution_type{},
                                           scaled_c_ij,
                                           state_type{});
                         }
                         const auto bounds = view.bounds(hd_i);

                         unsigned int next[1] = {
                             (unsigned int)((i + 1) % n_states)};
                         const auto P = 4.0 * (U_view.read_tensor(next) - U_i);
                         const auto [l, success] = view.limit(bounds, U_i, P);

                         dealii::Tensor<1, 4, double> result;
                         result[0] = bounds[0];
                         result[1] = bounds[1];
                         result[2] = bounds[2];
                         result[3] = l;
                         results_view.write_tensor(result, i);
                       });

  results.move_to_memory_space<HostSpace>();

  std::cout << "\nComputed on the device:\n";
  for (unsigned int i = 0; i < n_states; ++i) {
    const auto result = results.read_tensor<double>(i);
    std::cout << result[0] << " " << result[1] << " " << result[2] << " "
              << result[3] << "\n";
  }
}
