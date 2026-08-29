#include <gpu.h>

#include <deal.II/base/mpi.h>

#include <iostream>

/*
 * Test the array mode of ryujin::Mirrored, i.e., ryujin::Mirrored<T *>:
 * reinit(), size(), and read/write access via view() on both memory spaces.
 */


/* A "POD style" payload: */
struct Payload {
  double value;
  unsigned int index;
};

using HostSpace = dealii::MemorySpace::Host;
using DefaultSpace = dealii::MemorySpace::Default;
using ExecutionSpace = DefaultSpace::kokkos_space::execution_space;


void print_status(const ryujin::Mirrored<Payload *> &mirrored)
{
  std::cout << "size == " << mirrored.size() << std::endl
            << "HostSpace resident == " << mirrored.is_resident<HostSpace>()
            << std::endl
            << "DefaultSpace resident == "
            << mirrored.is_resident<DefaultSpace>() << std::endl;
}


void print_on_host_space(const ryujin::Mirrored<Payload *> &mirrored)
{
  const auto *payload = mirrored.view();

  std::cout << "Payload on HostSpace:";
  for (std::size_t i = 0; i < mirrored.size(); ++i)
    std::cout << " (" << payload[i].value << " " << payload[i].index << ")";
  std::cout << std::endl;
}


void print_on_default_space(const ryujin::Mirrored<Payload *> &mirrored)
{
  const auto *payload = mirrored.view<DefaultSpace>();
  const std::size_t n = mirrored.size();

  Kokkos::View<double *, DefaultSpace::kokkos_space> result("result", 2 * n);
  const auto exec = ExecutionSpace{};
  Kokkos::parallel_for(
      "mirrored_02",
      Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n),
      KOKKOS_LAMBDA(std::size_t i) {
        result(2 * i) = payload[i].value;
        result(2 * i + 1) = static_cast<double>(payload[i].index);
      });

  const auto result_host =
      Kokkos::create_mirror_view_and_copy(HostSpace::kokkos_space{}, result);

  std::cout << "Payload on DefaultSpace:";
  for (std::size_t i = 0; i < n; ++i)
    std::cout << " (" << result_host(2 * i) << " " << result_host(2 * i + 1)
              << ")";
  std::cout << std::endl;
}


int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  /*
   * An object constructed with the array mode constructor holds a zero
   * sized array: the storage is allocated (and resident) on the host
   * memory space.
   */

  ryujin::Mirrored<Payload *> mirrored("test payload array");

  std::cout << "After construction:" << std::endl;
  print_status(mirrored);

  std::cout << "\nAfter reinit(3):" << std::endl;
  mirrored.reinit(3);
  print_status(mirrored);

  {
    auto *payload = mirrored.view();
    for (std::size_t i = 0; i < mirrored.size(); ++i) {
      payload[i].value = 1. + double(i);
      payload[i].index = 10 + unsigned(i);
    }
  }

  print_on_host_space(mirrored);

  std::cout << "\nAfter copy to DefaultSpace:" << std::endl;
  mirrored.copy_to_memory_space<DefaultSpace>();
  print_status(mirrored);
  print_on_default_space(mirrored);

  std::cout << "\nAfter move to HostSpace:" << std::endl;
  mirrored.move_to_memory_space<HostSpace>();
  print_status(mirrored);

  /*
   * A reinit() with a new size and an implicit transfer policy. Requesting
   * a read only view on the default memory space copies the payload over.
   */

  std::cout << "\nAfter reinit(2) with implicit transfers:" << std::endl;
  mirrored.reinit(2, ryujin::TransferPolicy::implicit_transfers);
  print_status(mirrored);

  {
    auto *payload = mirrored.view();
    for (std::size_t i = 0; i < mirrored.size(); ++i) {
      payload[i].value = 0.5 * (1. + double(i));
      payload[i].index = 20 + unsigned(i);
    }
  }

  print_on_default_space(mirrored);
  print_status(mirrored);

  /* The zero length edge case: */

  std::cout << "\nAfter reinit(0):" << std::endl;
  mirrored.reinit(0);
  print_status(mirrored);
  mirrored.copy_to_memory_space<DefaultSpace>();
  print_status(mirrored);

  /*
   * A second object that is initialized on the default memory space (with
   * the defaulted last constructor argument) and that pins it with the
   * TransferPolicy::implicit_transfers_default_resident policy. The payload
   * is populated with a computation kernel.
   */

  std::cout << "\nA second object initialized on the DefaultSpace:"
            << std::endl;

  ryujin::Mirrored<Payload *> mirrored_2(
      "test payload array 2",
      2,
      ryujin::TransferPolicy::implicit_transfers_default_resident,
      DefaultSpace{});

  std::cout << "HostSpace pinned == " << mirrored_2.is_pinned<HostSpace>()
            << std::endl
            << "DefaultSpace pinned == " << mirrored_2.is_pinned<DefaultSpace>()
            << std::endl;
  print_status(mirrored_2);

  {
    auto *payload = mirrored_2.view<DefaultSpace>();
    const auto exec = ExecutionSpace{};
    Kokkos::parallel_for(
        "mirrored_02",
        Kokkos::RangePolicy<ExecutionSpace>(exec, 0, 2),
        KOKKOS_LAMBDA(std::size_t i) {
          payload[i].value = 100. + double(i);
          payload[i].index = 100 + unsigned(i);
        });
    Kokkos::fence();
  }

  /* A read only view on the host memory space copies the payload back: */

  print_on_host_space(mirrored_2);
  print_status(mirrored_2);

  /*
   * A reinit() of an object whose current transfer policy pins a memory
   * space: the policy is dropped for the duration of the reinit.
   */

  std::cout << "\nAfter reinit(1) of the second object:" << std::endl;
  mirrored_2.reinit(1, ryujin::TransferPolicy::implicit_transfers);
  print_status(mirrored_2);

  {
    auto *payload = mirrored_2.view();
    payload[0].value = 7.5;
    payload[0].index = 7;
  }

  print_on_default_space(mirrored_2);
  print_status(mirrored_2);
}
