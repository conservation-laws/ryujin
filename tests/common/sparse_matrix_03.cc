#include <sparse_matrix.h>

int main(int argc, char *argv[])
{
  //
  // Test creation of SparsityPatternView from mutable and const SparseMatrix:
  //

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  using VA = dealii::VectorizedArray<double>;
  constexpr auto simd_width = VA::size();

  /* Create sparsity pattern: */

  dealii::DynamicSparsityPattern dsp(1, 1);
  dsp.add(0, 0);
  dsp.compress();

  dealii::IndexSet locally_owned(1);
  locally_owned.add_range(0, 1);
  dealii::IndexSet locally_relevant(1);
  auto partitioner = std::make_shared<dealii::Utilities::MPI::Partitioner>(
      locally_owned, locally_relevant, MPI_COMM_SELF);

  ryujin::SparsityPattern<simd_width> sparsity_pattern(0, dsp, partitioner);

  ryujin::SparseMatrix<double, 1, simd_width> sparse_matrix;
  sparse_matrix.reinit(sparsity_pattern);


  auto view_1 = sparse_matrix.view();
  const auto view_2 = sparse_matrix.view();
  std::cout << view_1.read_entry(0, 0) << std::endl;
  std::cout << view_2.read_entry(0, 0) << std::endl;
  view_1.write_entry(0., 0, 0); // OK
  view_2.write_entry(1., 0, 0); // OK
  view_1.add_entry(2., 0, 0);   // OK
  view_2.add_entry(3., 0, 0);   // OK
  std::cout << view_1.read_entry(0, 0) << std::endl;
  std::cout << view_2.read_entry(0, 0) << std::endl;

  auto &sparse_matrix_ref_1 = sparse_matrix;

  auto view_3 = sparse_matrix_ref_1.view();
  const auto view_4 = sparse_matrix_ref_1.view();

  std::cout << view_3.read_entry(0, 0) << std::endl;
  std::cout << view_4.read_entry(0, 0) << std::endl;
  view_3.write_entry(10., 0, 0); // OK
  view_4.write_entry(11., 0, 0); // OK
  view_3.add_entry(2., 0, 0);    // OK
  view_4.add_entry(3., 0, 0);    // OK
  std::cout << view_3.read_entry(0, 0) << std::endl;
  std::cout << view_4.read_entry(0, 0) << std::endl;

  const auto &sparse_matrix_ref_2 = sparse_matrix;

  auto view_5 = sparse_matrix_ref_2.view();
  const auto view_6 = sparse_matrix_ref_2.view();

  std::cout << view_5.read_entry(0, 0) << std::endl;
  std::cout << view_6.read_entry(0, 0) << std::endl;
  // view_5.write_entry(0., 0, 0); // disallowed due to writable == false
  // view_6.write_entry(1., 0, 0); // disallowed due to writable == false
  // view_5.add_entry(2., 0, 0);   // disallowed due to writable == false
  // view_6.add_entry(3., 0, 0);   // disallowed due to writable == false
}
