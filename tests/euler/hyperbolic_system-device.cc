#include <hyperbolic_system.h>
#include <multicomponent_vector.h>

#include <iomanip>
#include <iostream>

int main(int argc, char *argv[])
{
  //
  // Test that the HyperbolicSystemView can be used on the device memory
  // space:
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

  const auto host_view = hyperbolic_system.view<dim, double, HostSpace>();
  const auto device_view = hyperbolic_system.view<dim, double, DefaultSpace>();

  using View = ryujin::Euler::HyperbolicSystemView<dim, double>;
  using state_type = typename View::state_type;

  std::cout << "gamma (host view) == " << host_view.gamma() << std::endl;
  std::cout << "gamma (device view) == " << device_view.gamma() << std::endl;

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

  /* Fill states on the host space: */

  for (unsigned int i = 0; i < n_states; ++i) {
    state_type primitive;
    primitive[0] = 1. + 0.125 * i;
    primitive[1] = 0.1 * i;
    primitive[2] = -0.05 * i;
    primitive[3] = 1. + 0.25 * i;
    const auto U_i = host_view.from_primitive_state(primitive);
    U.write_tensor(U_i, i);
  }

  /* Compute pressure, speed of sound, and energy flux on the host: */

  std::cout << "\nComputed on the host:\n";
  for (unsigned int i = 0; i < n_states; ++i) {
    const auto U_i = U.read_tensor<double>(i);
    std::cout << host_view.pressure(U_i) << " " << host_view.speed_of_sound(U_i)
              << " " << host_view.f(U_i)[dim + 1][0] << "\n";
  }

  /* Compute the same quantities on the default space: */

  U.move_to_memory_space<DefaultSpace>();

  ryujin::Vectors::MultiComponentVector<double, 3> results;
  results.reinit_with_scalar_partitioner(scalar_partitioner);
  results.move_to_memory_space<DefaultSpace>();

  const auto U_view = U.get_view<DefaultSpace>();
  const auto results_view = results.get_view<DefaultSpace>();

  using ExecutionSpace = DefaultSpace::kokkos_space::execution_space;
  const auto exec = ExecutionSpace{};
  Kokkos::parallel_for("test",
                       Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n_states),
                       [=](std::size_t i) {
                         const auto U_i = U_view.read_tensor(i);
                         dealii::Tensor<1, 3, double> result;
                         result[0] = device_view.pressure(U_i);
                         result[1] = device_view.speed_of_sound(U_i);
                         result[2] = device_view.f(U_i)[dim + 1][0];
                         results_view.write_tensor(result, i);
                       });

  results.move_to_memory_space<HostSpace>();

  std::cout << "\nComputed on the device:\n";
  for (unsigned int i = 0; i < n_states; ++i) {
    const auto result = results.read_tensor<double>(i);
    std::cout << result[0] << " " << result[1] << " " << result[2] << "\n";
  }
}
