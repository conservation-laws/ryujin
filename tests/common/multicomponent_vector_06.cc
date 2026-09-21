#include <multicomponent_vector.h>

int main(int argc, char *argv[])
{
  //
  // Test MultiComponentVectorView::extract_component() on both memory
  // spaces.
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
      ryujin::Vectors::create_vector_partitioner(scalar_partitioner, 4);

  ryujin::Vectors::MultiComponentVector<double, 4> state_vector;

  state_vector.reinit_with_vector_partitioner(vector_partitioner);

  using HostSpace = dealii::MemorySpace::Host;
  using DefaultSpace = dealii::MemorySpace::Default;

  for (unsigned int i = 0; i < 12; ++i) {
    dealii::Tensor<1, 4, double> tensor{{static_cast<double>(10 * i),
                                         static_cast<double>(10 * i + 1),
                                         static_cast<double>(10 * i + 2),
                                         static_cast<double>(10 * i + 3)}};
    state_vector.view().write_tensor<double>(tensor, i);
  }

  const auto print = [](const auto &vector) {
    for (unsigned int i = 0; i < 12; ++i)
      std::cout << vector.local_element(i) << (i == 11 ? "\n" : " ");
  };

  const auto times_two = [](const double value) { return 2. * value; };

  /* extract_component() on the host memory space: */

  dealii::LinearAlgebra::distributed::Vector<double, HostSpace> component;
  dealii::LinearAlgebra::distributed::Vector<double, HostSpace> scaled;
  component.reinit(scalar_partitioner);
  scaled.reinit(scalar_partitioner);

  {
    const auto view = state_vector.view<HostSpace>();
    view.extract_component(component, 2);
    view.extract_component(scaled, 1, times_two);
  }

  std::cout << "Component 2 extracted on HostSpace:\n";
  print(component);
  std::cout << "Component 1 extracted on HostSpace (scaled by 2):\n";
  print(scaled);

  /* extract_component() on the default memory space: */

  state_vector.move_to_memory_space<DefaultSpace>();

  dealii::LinearAlgebra::distributed::Vector<double, DefaultSpace>
      device_component;
  dealii::LinearAlgebra::distributed::Vector<double, DefaultSpace>
      device_scaled;
  device_component.reinit(scalar_partitioner);
  device_scaled.reinit(scalar_partitioner);

  {
    const auto view = state_vector.view<DefaultSpace>();
    view.extract_component(device_component, 2);
    view.extract_component(device_scaled, 1, times_two);
  }

  component.import_elements(device_component, dealii::VectorOperation::insert);
  scaled.import_elements(device_scaled, dealii::VectorOperation::insert);

  std::cout << "Component 2 extracted on DefaultSpace:\n";
  print(component);
  std::cout << "Component 1 extracted on DefaultSpace (scaled by 2):\n";
  print(scaled);
}
