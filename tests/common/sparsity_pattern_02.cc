#include <sparse_matrix.h>
#include <sparsity_pattern.h>

int main(int argc, char *argv[])
{
  //
  // Test the memory space residency of SparsityPattern: after reinit()
  // the pattern is resident on both memory spaces; creating a
  // SparseMatrixView for a memory space requires the pattern to be
  // resident there.
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

  using HostSpace = dealii::MemorySpace::Host;
  using DefaultSpace = dealii::MemorySpace::Default;

  const auto print_status = [&]() {
    std::cout << "HostSpace resident == "
              << sparsity_pattern.is_resident<HostSpace>() << std::endl
              << "DefaultSpace resident == "
              << sparsity_pattern.is_resident<DefaultSpace>() << std::endl;
  };

  /* After reinit the pattern is resident on both memory spaces: */

  print_status();

  /* Moving to the host deallocates the device mirror: */

  std::cout << "After move to HostSpace:" << std::endl;
  sparsity_pattern.move_to_memory_space<HostSpace>();
  print_status();

  /* Set up a sparse matrix on the (host resident) pattern: */

  ryujin::SparseMatrix<double, 1, simd_width> sparse_matrix;
  sparse_matrix.reinit(sparsity_pattern);

  {
    const auto host_view = sparse_matrix.view();
    host_view.write_entry(22.0, 0, 1);
    host_view.write_entry(20.0, 0, 2);
    host_view.write_entry(220.0, 1, 1);
    host_view.write_entry(200.0, 1, 2);
    host_view.write_entry(2200.0, 2, 1);
    host_view.write_entry(2000.0, 2, 2);
  }

  /*
   * Creating a device view of the sparse matrix requires a device
   * resident sparsity pattern - copy the pattern back over:
   */

  std::cout << "After copy to DefaultSpace:" << std::endl;
  sparsity_pattern.copy_to_memory_space<DefaultSpace>();
  print_status();

  /* Sum up rows on the default space: */

  sparse_matrix.move_to_memory_space<DefaultSpace>();

  const auto &view = sparse_matrix.template view<DefaultSpace>();
  using ExecutionSpace = DefaultSpace::kokkos_space::execution_space;
  const auto exec = ExecutionSpace{};
  Kokkos::parallel_for(
      "test",
      Kokkos::RangePolicy<ExecutionSpace>(exec, 0, 3),
      KOKKOS_LAMBDA(std::size_t i) {
        const auto a = view.read_entry(i, 1);
        const auto b = view.read_entry(i, 2);
        view.write_entry(a + b, i, 0);
      });
  exec.fence();

  /* Read entries on the host space: */

  sparse_matrix.move_to_memory_space<HostSpace>();

  const auto host_view = sparse_matrix.view();
  std::cout << "Entry (0, 0): " << host_view.read_entry(0, 0) << std::endl;
  std::cout << "Entry (1, 0): " << host_view.read_entry(1, 0) << std::endl;
  std::cout << "Entry (2, 0): " << host_view.read_entry(2, 0) << std::endl;
}
