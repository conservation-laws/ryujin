#include <loop.h>
#include <multicomponent_vector.h>

#include <iomanip>
#include <iostream>

//
// Test the reduction_loop() driver: We run the same loop bodies - generic,
// host/device capable lambdas that receive the vector view as forwarded
// argument and return their contribution - over the host memory space and
// over the default memory space and compare both results. The initial
// contents of the result storage take part in the reduction.
//

using namespace ryujin;

using HostSpace = dealii::MemorySpace::Host;
using DefaultSpace = dealii::MemorySpace::Default;

constexpr unsigned int n_states = 12;
constexpr int n_comp = 2;

int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  std::cout << std::setprecision(10);
  std::cout << std::scientific;

  /* Set up locally owned and relevant index sets. */

  dealii::IndexSet locally_owned(n_states);
  dealii::IndexSet locally_relevant(n_states);
  locally_owned.add_range(0, n_states);
  locally_relevant.add_range(0, n_states);

  const auto scalar_partitioner =
      std::make_shared<dealii::Utilities::MPI::Partitioner>(
          locally_owned, locally_relevant, MPI_COMM_WORLD);

  Vectors::MultiComponentVector<double, n_comp> U;
  U.reinit_with_scalar_partitioner(scalar_partitioner);

  for (unsigned int i = 0; i < n_states; ++i) {
    dealii::Tensor<1, n_comp, double> U_i;
    U_i[0] = 1. + 0.5 * i;
    U_i[1] = -2. + 0.25 * i;
    U.view().write_tensor<double>(U_i, i);
  }

  /* Scalar reductions: */

  const auto min_body =
      KOKKOS_LAMBDA(auto sentinel, const auto &U_view, unsigned int i)
  {
    using T = decltype(sentinel);
    const auto U_i = U_view.template read_tensor<T>(i);
    return U_i[1];
  };

  const auto sum_body =
      KOKKOS_LAMBDA(auto sentinel, const auto &U_view, unsigned int i)
  {
    using T = decltype(sentinel);
    const auto U_i = U_view.template read_tensor<T>(i);
    return U_i[0];
  };

  const auto run = [&](const std::string &name, auto memory_space) {
    using MemorySpace = decltype(memory_space);

    /* Initial values that are joined with the result of the loop: */
    double min_result = -1.5;
    double sum_result = 100.;

    U.template move_to_memory_space<MemorySpace>();
    const auto U_view = U.template view<MemorySpace>();

    reduction_loop<MemorySpace>("loop_02",
                                min_body,
                                Kokkos::Min<double>(min_result),
                                0,
                                n_states,
                                U_view);
    reduction_loop<MemorySpace>("loop_02",
                                sum_body,
                                Kokkos::Sum<double>(sum_result),
                                0,
                                n_states,
                                U_view);

    std::cout << name << ":\n";
    std::cout << "  min:  " << min_result << "\n";
    std::cout << "  sum:  " << sum_result << "\n";
  };

  std::cout << "Results for " << n_states << " states:\n";
  run("host", HostSpace{});
  run("device", DefaultSpace{});
}
