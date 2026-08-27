#include <sparse_matrix.h>
#include <sparsity_pattern.template.h>

int main(int argc, char *argv[])
{
  //
  // Test SparseMatrix access with a "warp size" (the row interval that the
  // sparsity pattern bins together) that is larger than the SIMD length
  // (the number of packed doubles the access operators work on).
  //

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  using VA = dealii::VectorizedArray<double>;
  constexpr unsigned int simd_length = VA::size();

  /*
   * A warp size of 16 is an integer multiple of every SIMD length that we
   * might encounter for a double (1, 2, 4, 8, or 16). This ensures that
   * the output of this test is independent of the vectorization variant
   * while exercising the warp_size > simd_length code paths.
   */
  constexpr int warp_size = 16;
  static_assert(warp_size % simd_length == 0, "test assumption");

  constexpr unsigned int n_dofs = 18;
  constexpr unsigned int n_internal_dofs = 16;

  /* A tridiagonal matrix with wrap around: every row has 3 entries. */

  dealii::DynamicSparsityPattern dsp(n_dofs, n_dofs);
  for (unsigned int i = 0; i < n_dofs; ++i) {
    dsp.add(i, i);
    dsp.add(i, (i + n_dofs - 1) % n_dofs);
    dsp.add(i, (i + 1) % n_dofs);
  }
  dsp.compress();

  dealii::IndexSet locally_owned(n_dofs);
  locally_owned.add_range(0, n_dofs);
  dealii::IndexSet locally_relevant(n_dofs);
  auto partitioner = std::make_shared<dealii::Utilities::MPI::Partitioner>(
      locally_owned, locally_relevant, MPI_COMM_SELF);

  ryujin::SparsityPattern<warp_size> sparsity_pattern(
      n_internal_dofs, dsp, partitioner);

  ryujin::SparseMatrix<double, 1, warp_size> sparse_matrix(sparsity_pattern);

  const auto pattern_view = sparsity_pattern.view();
  const auto matrix_view = sparse_matrix.view();

  std::cout << "Stride of row 0:  " << pattern_view.stride_of_row(0)
            << std::endl;
  std::cout << "Stride of row 17: " << pattern_view.stride_of_row(17)
            << std::endl;

  /*
   * Populate the matrix with the values 10 * row + column_index. In the
   * internal index range we write vectorized, i.e., stepping forward with
   * the SIMD length and not with the warp size:
   */

  const auto value = [](const unsigned int row, const unsigned int col_idx) {
    return 10. * row + col_idx;
  };

  for (unsigned int i = 0; i < n_internal_dofs; i += simd_length) {
    for (unsigned int col_idx = 0; col_idx < 3; ++col_idx) {
      VA entry;
      for (unsigned int k = 0; k < simd_length; ++k)
        entry[k] = value(i + k, col_idx);
      matrix_view.template write_entry<VA>(entry, i, col_idx);
    }
  }

  for (unsigned int i = n_internal_dofs; i < n_dofs; ++i)
    for (unsigned int col_idx = 0; col_idx < 3; ++col_idx)
      matrix_view.write_entry(value(i, col_idx), i, col_idx);

  /* Read back scalar: */

  std::cout << "Matrix entries row by row" << std::endl;
  for (unsigned int i = 0; i < pattern_view.n_rows(); ++i) {
    for (unsigned int col_idx = 0; col_idx < pattern_view.row_length(i);
         ++col_idx)
      std::cout << matrix_view.read_entry(i, col_idx) << " ";
    std::cout << std::endl;
  }

  std::cout << "Matrix entries transposed row by row" << std::endl;
  for (unsigned int i = 0; i < pattern_view.n_rows(); ++i) {
    for (unsigned int col_idx = 0; col_idx < pattern_view.row_length(i);
         ++col_idx)
      std::cout << matrix_view.read_transposed_entry(i, col_idx) << " ";
    std::cout << std::endl;
  }

  /*
   * Verify that the transposed entries are what we expect them to be,
   * i.e., that (i, col_idx) -> (j, column index of i in row j):
   */

  bool transposed_consistent = true;
  for (unsigned int i = 0; i < pattern_view.n_rows(); ++i) {
    const unsigned int stride_size = pattern_view.stride_of_row(i);
    const unsigned int *js = pattern_view.columns(i);
    for (unsigned int col_idx = 0; col_idx < pattern_view.row_length(i);
         ++col_idx) {
      const auto j = js[col_idx * stride_size];
      const auto expected = value(j, pattern_view.column_index(j, i));
      if (matrix_view.read_transposed_entry(i, col_idx) != expected)
        transposed_consistent = false;
    }
  }
  std::cout << "Transposed entries consistent: " << transposed_consistent
            << std::endl;

  /*
   * Verify that vectorized access returns the same values as scalar
   * access. We only print a boolean so that the output of this test does
   * not depend on the SIMD length:
   */

  bool vectorized_consistent = true;
  for (unsigned int i = 0; i < n_internal_dofs; i += simd_length) {
    for (unsigned int col_idx = 0; col_idx < 3; ++col_idx) {
      const auto entry = matrix_view.template read_entry<VA>(i, col_idx);
      const auto transposed =
          matrix_view.template read_transposed_entry<VA>(i, col_idx);
      for (unsigned int k = 0; k < simd_length; ++k) {
        if (entry[k] != matrix_view.read_entry(i + k, col_idx))
          vectorized_consistent = false;
        if (transposed[k] != matrix_view.read_transposed_entry(i + k, col_idx))
          vectorized_consistent = false;
      }
    }
  }
  std::cout << "Vectorized access consistent: " << vectorized_consistent
            << std::endl;
}
