#include <multicomponent_vector.h>

int main(int argc, char *argv[])
{
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

  ryujin::Vectors::MultiComponentVector<double, 1> scalar_vector;

  scalar_vector.reinit_with_scalar_partitioner(scalar_partitioner);

  ryujin::Vectors::MultiComponentVector<double, 4> state_vector;

  state_vector.reinit_with_vector_partitioner(vector_partitioner);

  for (unsigned int i = 0; i < 12; ++i) {
    scalar_vector.write_entry<double>(static_cast<double>(i), i);

    dealii::Tensor<1, 4, double> tensor{{static_cast<double>(10 * i),
                                         static_cast<double>(10 * i + 1),
                                         static_cast<double>(10 * i + 2),
                                         static_cast<double>(10 * i + 3)}};
    state_vector.write_tensor<double>(tensor, i);
  }

  using VA = dealii::VectorizedArray<double>;
  constexpr auto simd_width = VA::size();

  std::cout << "Scalar vector (packed SIMD)\n";
  for (unsigned int i = 0; i < 8; i += simd_width) {
    std::cout << scalar_vector.read_entry<VA>(i) << "\n";
  }

  std::cout << "\nState vector (packed SIMD):\n";
  for (unsigned int i = 0; i < 8; i += simd_width) {
    std::cout << state_vector.read_tensor<VA>(i) << "\n";
  }
}
