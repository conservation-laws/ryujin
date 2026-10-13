#include <multicomponent_vector.h>

int main(int argc, char *argv[])
{
  //
  // Test sadd() on the underlying deal.II vectors of a
  // MultiComponentVector on both memory spaces.
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
      ryujin::create_vector_partitioner(scalar_partitioner, 4);

  ryujin::MultiComponentVector<double> state_vector;
  ryujin::MultiComponentVector<double> increment_vector;

  state_vector.reinit_with_vector_partitioner(vector_partitioner, 4);
  increment_vector.reinit_with_vector_partitioner(vector_partitioner, 4);

  using HostSpace = dealii::MemorySpace::Host;
  using DefaultSpace = dealii::MemorySpace::Default;

  for (unsigned int i = 0; i < 12; ++i) {
    dealii::Tensor<1, 4, double> tensor{{static_cast<double>(10 * i),
                                         static_cast<double>(10 * i + 1),
                                         static_cast<double>(10 * i + 2),
                                         static_cast<double>(10 * i + 3)}};
    state_vector.view<4>().write_tensor<double>(tensor, i);

    dealii::Tensor<1, 4, double> increment{{1., 2., 3., 4.}};
    increment_vector.view<4>().write_tensor<double>(increment, i);
  }

  const auto print = [](const auto &vector) {
    for (unsigned int i = 0; i < 4; ++i)
      std::cout << vector.template view<4>().template read_tensor<double>(i)
                << "\n";
  };

  /* sadd() on the host memory space: */

  {
    auto &vector = state_vector.deal_ii_vector<HostSpace>();
    const auto &increment = std::as_const(increment_vector).deal_ii_vector();
    vector.sadd(2., 3., increment);
  }

  std::cout << "After sadd() on HostSpace:\n";
  print(state_vector);

  /* sadd() on the default memory space: */

  state_vector.move_to_memory_space<DefaultSpace>();
  increment_vector.move_to_memory_space<DefaultSpace>();

  {
    auto &vector = state_vector.deal_ii_vector<DefaultSpace>();
    const auto &increment =
        std::as_const(increment_vector).deal_ii_vector<DefaultSpace>();
    vector.sadd(1., 2., increment);
  }

  state_vector.move_to_memory_space<HostSpace>();
  increment_vector.move_to_memory_space<HostSpace>();

  std::cout << "After sadd() on DefaultSpace:\n";
  print(state_vector);

  /* The second operand is left unmodified: */

  std::cout << "Increment vector:\n";
  print(increment_vector);
}
