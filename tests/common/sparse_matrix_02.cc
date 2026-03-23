#include <sparse_matrix.h>

int main(int argc, char *argv[])
{
  //
  // Test memory space transfer:
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
    std::cout << "HostSpace active == "
              << sparse_matrix.is_active_memory_space<HostSpace>() << std::endl
              << "DefaultSpace active == "
              << sparse_matrix.is_active_memory_space<DefaultSpace>()
              << std::endl;
  };

  /* Fill entries on the host space: */

  print_status();
  sparse_matrix.write_entry(22.0, 0, 1);
  sparse_matrix.write_entry(20.0, 0, 2);
  sparse_matrix.write_entry(220.0, 1, 1);
  sparse_matrix.write_entry(200.0, 1, 2);
  sparse_matrix.write_entry(2200.0, 2, 1);
  sparse_matrix.write_entry(2000.0, 2, 2);

  /* Sum up rows on the default space: */

  std::cout << "After move to DefaultSpace:" << std::endl;
  sparse_matrix.move_to_memory_space<DefaultSpace>();
  print_status();
  std::cout << "After repeated move to DefaultSpace:" << std::endl;
  sparse_matrix.move_to_memory_space<DefaultSpace>();
  print_status();

  const auto &view = sparse_matrix.template get_view<DefaultSpace>();
  using ExecutionSpace = DefaultSpace::kokkos_space::execution_space;
  const auto exec = ExecutionSpace{};
  Kokkos::parallel_for("test",
                       Kokkos::RangePolicy<ExecutionSpace>(exec, 0, 3),
                       [=](std::size_t i) {
                         const auto a = view.read_entry(i, 1);
                         const auto b = view.read_entry(i, 2);
                         view.write_entry(a + b, i, 0);
                       });


  /* Read entries on the host space: */

  std::cout << "After move to HostSpace:" << std::endl;
  sparse_matrix.move_to_memory_space<HostSpace>();
  print_status();

  std::cout << "Entry (0, 0): " << sparse_matrix.read_entry(0, 0) << std::endl;
  std::cout << "Entry (1, 1): " << sparse_matrix.read_entry(1, 0) << std::endl;
  std::cout << "Entry (2, 2): " << sparse_matrix.read_entry(2, 0) << std::endl;
}
