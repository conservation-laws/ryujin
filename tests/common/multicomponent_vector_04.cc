#include <multicomponent_vector.h>

int main(int argc, char *argv[])
{
  //
  // Test the TransferPolicy::implicit_transfers policy: a read-only view
  // triggers an implicit copy_to_memory_space(), a writable view triggers
  // an implicit move_to_memory_space() invalidating the other memory
  // space.
  //

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  /* Set up locally owned and relevant index sets. */

  dealii::IndexSet locally_owned(12);
  dealii::IndexSet locally_relevant(12);
  locally_owned.add_range(0, 12);
  locally_relevant.add_range(0, 12);

  const auto scalar_partitioner =
      std::make_shared<dealii::Utilities::MPI::Partitioner>(
          locally_owned, locally_relevant, MPI_COMM_WORLD);

  const auto vector_partitioner =
      ryujin::Vectors::create_vector_partitioner(scalar_partitioner, 4);

  ryujin::Vectors::MultiComponentVector<double, 4> state_vector;

  state_vector.reinit_with_vector_partitioner(
      vector_partitioner, ryujin::TransferPolicy::implicit_transfers);

  using HostSpace = dealii::MemorySpace::Host;
  using DefaultSpace = dealii::MemorySpace::Default;

  const auto print_status = [&]() {
    std::cout << "HostSpace resident == "
              << state_vector.is_resident<HostSpace>() << std::endl
              << "DefaultSpace resident == "
              << state_vector.is_resident<DefaultSpace>() << std::endl;
  };

  print_status();

  for (unsigned int i = 0; i < 12; ++i) {
    dealii::Tensor<1, 4, double> tensor{{static_cast<double>(10 * i),
                                         static_cast<double>(10 * i + 1),
                                         static_cast<double>(10 * i + 2),
                                         static_cast<double>(10 * i + 3)}};
    state_vector.view().write_tensor<double>(tensor, i);
  }

  /* A read-only view performs an implicit copy: */

  const auto &const_vector = state_vector;
  {
    [[maybe_unused]] const auto view =
        const_vector.template view<DefaultSpace>();
  }
  std::cout << "After read-only view<DefaultSpace>:" << std::endl;
  print_status();

  /* A writable view performs an implicit move invalidating the host: */

  {
    const auto view = state_vector.template view<DefaultSpace>();
    std::cout << "After writable view<DefaultSpace>:" << std::endl;
    print_status();

    using ExecutionSpace = DefaultSpace::kokkos_space::execution_space;
    const auto exec = ExecutionSpace{};
    Kokkos::parallel_for(
        "test",
        Kokkos::RangePolicy<ExecutionSpace>(exec, 0, 3),
        KOKKOS_LAMBDA(std::size_t i) {
          auto a = view.read_tensor(i);
          a *= 10.;
          view.write_tensor(a, i);
        });
    exec.fence();
  }

  /*
   * A read-only host view performs an implicit copy back:
   */

  {
    const auto view = const_vector.template view<HostSpace>();
    std::cout << "After read-only view<HostSpace>:" << std::endl;
    print_status();

    std::cout << "\nState vector:\n";
    for (unsigned int i = 0; i < 8; i += 1) {
      std::cout << view.template read_tensor<double>(i) << "\n";
    }
  }

  /* A writable host view invalidates the device mirror: */

  {
    [[maybe_unused]] const auto view = state_vector.template view<HostSpace>();
  }
  std::cout << "After writable view<HostSpace>:" << std::endl;
  print_status();

  /* Read the data back through a host view: */

  std::cout << "\nState vector:\n";
  for (unsigned int i = 0; i < 8; i += 1) {
    std::cout << state_vector.view().read_tensor<double>(i) << "\n";
  }
}
