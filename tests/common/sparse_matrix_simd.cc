#include <sparse_matrix_simd.h>
#include <sparse_matrix_simd.template.h>

int main()
{
  using VA = dealii::VectorizedArray<double>;
  constexpr auto simd_width = VA::size();

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
  auto partitioner = std::make_shared<dealii::Utilities::MPI::Partitioner>(
      locally_owned, locally_relevant, MPI_COMM_SELF);

  ryujin::SparsityPatternSIMD<simd_width> my_sparsity(
      (12 / simd_width) * simd_width, spars, partitioner);
  ryujin::SparseMatrixSIMD<double, 1, simd_width> my_sparse(my_sparsity);
  for (unsigned i = 0; i < 12; ++i)
    for (unsigned j = 0; j < 3; ++j)
      my_sparse.write_entry(double(i * 3 + j), i, j);
  my_sparse.write_entry(36., 12, 0);
  my_sparse.write_entry(37., 12, 1);
  my_sparse.write_entry(38., 13, 0);
  my_sparse.write_entry(39., 13, 1);
  std::cout << "Matrix entries row by row" << std::endl;
  for (unsigned int i = 0; i < my_sparsity.n_rows(); ++i) {
    for (unsigned int j = 0; j < my_sparsity.row_length(i); ++j) {
      const auto a = my_sparse.get_entry(i, j);
      std::cout << a << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "Matrix entries by SIMD rows" << std::endl;
  unsigned int i = 0;
  for (; i < (12 / simd_width) * simd_width; i += simd_width) {
    for (unsigned int j = 0; j < 3; ++j) {
      const auto a = my_sparse.template get_entry<VA>(i, j);
      std::cout << a << "   ";
    }
    std::cout << std::endl;
  }
  for (; i < 14; i++)
    std::cout << my_sparse.get_entry(i, 0) << " " << my_sparse.get_entry(i, 1)
              << " ";
  std::cout << std::endl;

  std::cout << "Matrix entries transposed row by row" << std::endl;
  for (unsigned int i = 0; i < my_sparsity.n_rows(); ++i) {
    for (unsigned int j = 0; j < my_sparsity.row_length(i); ++j) {
      const auto a = my_sparse.get_transposed_entry(i, j);
      std::cout << a << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "Matrix entries transposed by SIMD row" << std::endl;
  i = 0;
  for (; i < (12 / simd_width) * simd_width; i += simd_width) {
    for (unsigned int j = 0; j < 3; ++j) {
      const auto a = my_sparse.template get_transposed_entry<VA>(i, j);
      std::cout << a << "   ";
    }
    std::cout << std::endl;
  }
  for (; i < 14; i++)
    std::cout << my_sparse.get_transposed_entry(i, 0) << " "
              << my_sparse.get_transposed_entry(i, 1) << " ";
  std::cout << std::endl;
}
