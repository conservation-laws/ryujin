#include <gpu.h>

#include <deal.II/base/mpi.h>

#include <iostream>

/*
 * Test ryujin::Mirrored: residency handling and read/write access via
 * view() on both memory spaces.
 */


/* A "POD style" payload: */
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

  /*
   * The TransferPolicy::implicit_transfers_host_resident policy: The host
   * memory space is pinned, i.e., all operations that would deallocate the
   * host storage are disallowed. The pointer returned by view() on the
   * host memory space thus remains valid.
   */

  std::cout << "\nUsing TransferPolicy::implicit_transfers_host_resident:"
            << std::endl;

  ryujin::Mirrored<Parameters> mirrored_3(
      "test parameters 3",
      ryujin::TransferPolicy::implicit_transfers_host_resident);

  std::cout << "HostSpace pinned == " << mirrored_3.is_pinned<HostSpace>()
            << std::endl
            << "DefaultSpace pinned == " << mirrored_3.is_pinned<DefaultSpace>()
            << std::endl;

  Parameters *host_pointer = mirrored_3.view();
  host_pointer->gamma = 4.5;
  host_pointer->gamma_inverse = 1. / 4.5;
  host_pointer->n_iterations = 4;

  print_on_default_space(mirrored_3);
  print_status(mirrored_3);

  /*
   * A writable view on the host memory space still drops the (now stale)
   * mirror in the default memory space, but it never touches the host
   * storage:
   */

  mirrored_3.view()->n_iterations = 5;

  std::cout << "After a writable view on HostSpace:" << std::endl;
  print_status(mirrored_3);
  std::cout << "HostSpace pointer stable == "
            << (mirrored_3.view() == host_pointer) << std::endl;
  print_on_default_space(mirrored_3);
  print_status(mirrored_3);

  /*
   * The TransferPolicy::implicit_transfers_default_resident policy is the
   * converse: the default memory space is pinned. A writable view on the
   * host memory space is disallowed under this policy, so we populate the
   * payload first and select the transfer policy afterwards.
   */

  std::cout << "\nUsing TransferPolicy::implicit_transfers_default_resident:"
            << std::endl;

  ryujin::Mirrored<Parameters> mirrored_4("test parameters 4");

  {
    auto *parameters = mirrored_4.view();
    parameters->gamma = 5.5;
    parameters->gamma_inverse = 1. / 5.5;
    parameters->n_iterations = 6;
  }

  /*
   * The policy pins the default memory space, so the payload has to be put
   * in place before the policy is selected:
   */
  mirrored_4.copy_to_memory_space<DefaultSpace>();
  mirrored_4.set_transfer_policy(
      ryujin::TransferPolicy::implicit_transfers_default_resident);

  std::cout << "HostSpace pinned == " << mirrored_4.is_pinned<HostSpace>()
            << std::endl
            << "DefaultSpace pinned == " << mirrored_4.is_pinned<DefaultSpace>()
            << std::endl;

  /* The default memory space is resident, a read only view is a no-op: */

  const auto &const_mirrored_4 = mirrored_4;
  const Parameters *default_pointer = const_mirrored_4.view<DefaultSpace>();

  std::cout << "After a read only view on DefaultSpace:" << std::endl;
  print_status(mirrored_4);

  /* A writable view on the default memory space drops the host mirror: */

  auto *writable_pointer = mirrored_4.view<DefaultSpace>();

  std::cout << "After a writable view on DefaultSpace:" << std::endl;
  print_status(mirrored_4);
  std::cout << "DefaultSpace pointer stable == "
            << (writable_pointer == default_pointer) << std::endl;
  print_on_default_space(mirrored_4);

  /* A read only view on the host memory space copies the payload back: */

  print_on_host_space(mirrored_4);
  print_status(mirrored_4);
}
