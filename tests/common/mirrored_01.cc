#include <gpu.h>

#include <deal.II/base/mpi.h>

#include <iostream>

/*
 * A "POD style" payload:
 */
struct Parameters {
  double gamma;
  double gamma_inverse;
  unsigned int n_iterations;
};

using HostSpace = dealii::MemorySpace::Host;
using DefaultSpace = dealii::MemorySpace::Default;
using ExecutionSpace = DefaultSpace::kokkos_space::execution_space;


void print_status(const ryujin::Mirrored<Parameters> &mirrored)
{
  std::cout << "HostSpace resident == " << mirrored.is_resident<HostSpace>()
            << std::endl
            << "DefaultSpace resident == "
            << mirrored.is_resident<DefaultSpace>() << std::endl;
}


void print_on_host_space(const ryujin::Mirrored<Parameters> &mirrored)
{
  const auto *parameters = mirrored.view();

  std::cout << "Parameters on HostSpace: " << parameters->gamma << " "
            << parameters->gamma_inverse << " " << parameters->n_iterations
            << std::endl;
}


/*
 * Read the parameters back on the default memory space.
 */
void print_on_default_space(const ryujin::Mirrored<Parameters> &mirrored)
{
  const auto *parameters = mirrored.view<DefaultSpace>();

  Kokkos::View<double *, DefaultSpace::kokkos_space> result("result", 3);
  const auto exec = ExecutionSpace{};
  Kokkos::parallel_for("mirrored_01",
                       Kokkos::RangePolicy<ExecutionSpace>(exec, 0, 1),
                       [=](std::size_t) {
                         result(0) = parameters->gamma;
                         result(1) = parameters->gamma_inverse;
                         result(2) =
                             static_cast<double>(parameters->n_iterations);
                       });

  const auto result_host =
      Kokkos::create_mirror_view_and_copy(HostSpace::kokkos_space{}, result);

  std::cout << "Parameters on DefaultSpace: " << result_host(0) << " "
            << result_host(1) << " " << result_host(2) << std::endl;
}


int main(int argc, char *argv[])
{
  //
  // Test ryujin::Mirrored: residency handling and read/write access via
  // view() on both memory spaces.
  //

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  ryujin::Mirrored<Parameters> mirrored("test parameters");

  std::cout << "After construction:" << std::endl;
  print_status(mirrored);

  {
    auto *parameters = mirrored.view();
    parameters->gamma = 1.5;
    parameters->gamma_inverse = 1. / 1.5;
    parameters->n_iterations = 42;
  }

  std::cout << std::endl;
  print_on_host_space(mirrored);

  std::cout << "\nAfter copy to DefaultSpace:" << std::endl;
  mirrored.copy_to_memory_space<DefaultSpace>();
  print_status(mirrored);
  print_on_default_space(mirrored);

  /*
   * A runtime parameter change: Drop the (stale) mirror in the default
   * memory space, update the parameters on the host, and push the update
   * over again.
   */

  std::cout << "\nAfter move to HostSpace:" << std::endl;
  mirrored.move_to_memory_space<HostSpace>();
  print_status(mirrored);

  {
    auto *parameters = mirrored.view();
    parameters->gamma = 2.5;
    parameters->gamma_inverse = 1. / 2.5;
  }

  std::cout << "\nAfter copy to DefaultSpace:" << std::endl;
  mirrored.copy_to_memory_space<DefaultSpace>();
  print_status(mirrored);
  print_on_default_space(mirrored);

  /*
   * The same round trip with implicit transfers: Requesting a writable
   * view on the host drops the mirror in the default memory space, and
   * requesting a read only view on the default memory space restores it.
   */

  std::cout << "\nUsing TransferPolicy::implicit_transfers:" << std::endl;
  mirrored.set_transfer_policy(ryujin::TransferPolicy::implicit_transfers);

  mirrored.view()->n_iterations = 7;

  std::cout << "After a writable view on HostSpace:" << std::endl;
  print_status(mirrored);
  print_on_host_space(mirrored);

  std::cout << "After a read only view on DefaultSpace:" << std::endl;
  print_on_default_space(mirrored);
  print_status(mirrored);

  /*
   * The transfer policy can also be selected at construction time:
   */

  std::cout << "\nA second object constructed with implicit transfers:"
            << std::endl;

  ryujin::Mirrored<Parameters> mirrored_2(
      "test parameters 2", ryujin::TransferPolicy::implicit_transfers);

  {
    auto *parameters = mirrored_2.view();
    parameters->gamma = 3.5;
    parameters->gamma_inverse = 1. / 3.5;
    parameters->n_iterations = 3;
  }

  print_status(mirrored_2);
  print_on_default_space(mirrored_2);
  print_status(mirrored_2);
}
