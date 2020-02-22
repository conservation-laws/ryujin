#include <sparse_matrix_simd.h>

int main()
{
  dealii::DynamicSparsityPattern spars(14, 14);
  spars.add(0, 0);
  spars.add(0, 1);
  spars.add(0, 13);
  for (unsigned int i = 1; i < 12; ++i) {
    spars.add(i, i - 1);
    spars.add(i, i);
    spars.add(i, i + 1);
  }
  spars.add(12, 12);
  spars.add(12, 11);
  spars.add(13, 13);
  spars.add(13, 0);
  spars.compress();

  dealii::IndexSet locally_owned(14);
  locally_owned.add_range(0, 14);
  dealii::IndexSet locally_relevant(14);
  dealii::Utilities::MPI::Partitioner partitioner(
      locally_owned, locally_relevant, MPI_COMM_SELF);

  grendel::SparsityPatternSIMD<4> my_sparsity(12, spars,partitioner);
  grendel::SparseMatrixSIMD<double, 1, 4> my_sparse(my_sparsity);
  for (unsigned i = 0; i < 12; ++i)
    for (unsigned j = 0; j < 3; ++j)
      my_sparse.write_entry(i * 3 + j, i, j);
  my_sparse.write_entry(36, 12, 0);
  my_sparse.write_entry(37, 12, 1);
  my_sparse.write_entry(38, 13, 0);
  my_sparse.write_entry(39, 13, 1);
  std::cout << "Matrix entries row by row" << std::endl;
  for (unsigned int i = 0; i < my_sparsity.n_rows(); ++i) {
    for (unsigned int j = 0; j < my_sparsity.row_length(i); ++j) {
      const auto a = my_sparse.get_entry(i, j);
      std::cout << a << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "Matrix entries by SIMD rows" << std::endl;
  for (unsigned int i = 0; i < 12; i += 4) {
    for (unsigned int j = 0; j < 3; ++j) {
      const auto a = my_sparse.get_vectorized_entry(i, j);
      std::cout << a << "   ";
    }
    std::cout << std::endl;
  }
  std::cout << my_sparse.get_entry(12, 0) << " " << my_sparse.get_entry(12, 1)
            << " " << my_sparse.get_entry(13, 0) << " "
            << my_sparse.get_entry(13, 1) << std::endl;

  std::cout << "Matrix entries transposed row by row" << std::endl;
  for (unsigned int i = 0; i < my_sparsity.n_rows(); ++i) {
    for (unsigned int j = 0; j < my_sparsity.row_length(i); ++j) {
      const auto a = my_sparse.get_transposed_entry(i, j);
      std::cout << a << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "Matrix entries transposed by SIMD row" << std::endl;
  for (unsigned int i = 0; i < 12; i += 4) {
    for (unsigned int j = 0; j < 3; ++j) {
      const auto a = my_sparse.get_vectorized_transposed_entry(i, j);
      std::cout << a << "   ";
    }
    std::cout << std::endl;
  }
  std::cout << my_sparse.get_transposed_entry(12, 0) << " "
            << my_sparse.get_transposed_entry(12, 1) << " "
            << my_sparse.get_transposed_entry(13, 0) << " "
            << my_sparse.get_transposed_entry(13, 1) << std::endl;
}
