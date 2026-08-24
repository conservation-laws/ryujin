#include <sparse_matrix.h>

int main(int argc, char *argv[])
{
  //
  // Test the MirroredStorage state machine with explicit transfers:
  // copy_to_memory_space() keeps both memory spaces resident,
  // move_to_memory_space() deallocates the moved-from memory space.
  //

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  using VA = dealii::VectorizedArray<double>;
  constexpr auto simd_width = VA::size();

  /* Create sparsity pattern: */

  dealii::DynamicSparsityPattern dsp(3, 3);
  dsp.add(0, 0);
  dsp.add(0, 1);
  dsp.add(0, 2);
  dsp.add(1, 0);
  dsp.add(1, 1);
  dsp.add(1, 2);
  dsp.add(2, 0);
  dsp.add(2, 1);
  dsp.add(2, 2);
  dsp.compress();

  dealii::IndexSet locally_owned(3);
  locally_owned.add_range(0, 3);
  dealii::IndexSet locally_relevant(3);
  auto partitioner = std::make_shared<dealii::Utilities::MPI::Partitioner>(
      locally_owned, locally_relevant, MPI_COMM_SELF);

  ryujin::SparsityPattern<simd_width> sparsity_pattern(0, dsp, partitioner);

  ryujin::SparseMatrix<double, 1, simd_width> sparse_matrix;
  sparse_matrix.reinit(sparsity_pattern);

  using HostSpace = dealii::MemorySpace::Host;
  using DefaultSpace = dealii::MemorySpace::Default;

  const auto print_status = [&]() {
    std::cout << "HostSpace resident == "
              << sparse_matrix.is_resident<HostSpace>() << std::endl
              << "DefaultSpace resident == "
              << sparse_matrix.is_resident<DefaultSpace>() << std::endl;
  };

  /* Fill entries on the host space: */

  print_status();
  sparse_matrix.write_entry(22.0, 0, 1);
  sparse_matrix.write_entry(20.0, 0, 2);
  sparse_matrix.write_entry(220.0, 1, 1);
  sparse_matrix.write_entry(200.0, 1, 2);
  sparse_matrix.write_entry(2200.0, 2, 1);
  sparse_matrix.write_entry(2000.0, 2, 2);

  /* Copying to the default space keeps both spaces resident: */

  std::cout << "After copy to DefaultSpace:" << std::endl;
  sparse_matrix.copy_to_memory_space<DefaultSpace>();
  print_status();
  std::cout << "After repeated copy to DefaultSpace:" << std::endl;
  sparse_matrix.copy_to_memory_space<DefaultSpace>();
  print_status();

  /*
   * Under explicit transfers a writable host access with both memory
   * spaces resident is permitted and performs no invalidation:
   */

  {
    auto view = sparse_matrix.view<HostSpace>();
    view.write_entry(4.0, 0, 1);
  }
  std::cout << "After writable host access:" << std::endl;
  print_status();

  /*
   * Moving to the host with both memory spaces resident deallocates the
   * device mirror without copying it back - the host mutation above must
   * survive:
   */

  std::cout << "After move to HostSpace:" << std::endl;
  sparse_matrix.move_to_memory_space<HostSpace>();
  print_status();
  std::cout << "Entry (0, 1): " << sparse_matrix.read_entry(0, 1) << std::endl;

  /* Sum up rows on the default space: */

  std::cout << "After move to DefaultSpace:" << std::endl;
  sparse_matrix.move_to_memory_space<DefaultSpace>();
  print_status();

  const auto &view = sparse_matrix.template view<DefaultSpace>();
  using ExecutionSpace = DefaultSpace::kokkos_space::execution_space;
  const auto exec = ExecutionSpace{};
  Kokkos::parallel_for("test",
                       Kokkos::RangePolicy<ExecutionSpace>(exec, 0, 3),
                       [=](std::size_t i) {
                         const auto a = view.read_entry(i, 1);
                         const auto b = view.read_entry(i, 2);
                         view.write_entry(a + b, i, 0);
                       });
  exec.fence();

  /* Copying back to the host keeps the device data resident: */

  std::cout << "After copy to HostSpace:" << std::endl;
  sparse_matrix.copy_to_memory_space<HostSpace>();
  print_status();

  /*
   * Note: After a (logically const) copy to the host the inherited
   * direct-access interface remains detached, so we read through a
   * freshly created view:
   */

  const auto &const_matrix = sparse_matrix;
  const auto host_view = const_matrix.view<HostSpace>();
  std::cout << "Entry (0, 0): " << host_view.read_entry(0, 0) << std::endl;
  std::cout << "Entry (1, 0): " << host_view.read_entry(1, 0) << std::endl;
  std::cout << "Entry (2, 0): " << host_view.read_entry(2, 0) << std::endl;
}
