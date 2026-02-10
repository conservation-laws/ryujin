#include <multicomponent_vector.h>

int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  /* Set up locally owned and relevant index sets. */

  dealii::IndexSet locally_owned(16);
  dealii::IndexSet locally_relevant(16);

  locally_owned.add_range(0, 8);
  locally_relevant.add_range(0, 8);
  locally_relevant.add_range(12, 16);

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

  std::cout << "Scalar vector:\n";
  for (unsigned int i = 0; i < 12; ++i) {
    std::cout << scalar_vector.get_entry<double>(i) << "\n";
  }

  std::cout << "\nState vector:\n";
  for (unsigned int i = 0; i < 12; ++i) {
    std::cout << state_vector.get_tensor<double>(i) << "\n";
  }
}
