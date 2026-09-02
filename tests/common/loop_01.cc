#include <loop.h>
#include <multicomponent_vector.h>

#include <array>
#include <iomanip>
#include <iostream>

//
// Test the gpu_loop() driver: We run the same loop body - a generic,
// host/device capable lambda that receives the vector views as forwarded
// arguments - over the host memory space with cpu_simd_loop() and over the
// default memory space with gpu_loop() and compare both results.
//

using namespace ryujin;

using HostSpace = dealii::MemorySpace::Host;
using DefaultSpace = dealii::MemorySpace::Default;

constexpr unsigned int n_states = 12;
constexpr int n_comp = 2;

/*
 * nvcc does not allow a type local to a function as template argument of a
 * function that instantiates an extended __host__ __device__ lambda. The
 * loop body therefore has to be a functor defined at namespace scope.
 */
struct Body {
  template <typename T, typename ReadView, typename WriteView>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
  operator()(T /*sentinel*/,
             const ReadView &U_view,
             const WriteView &results_view,
             unsigned int i) const
  {
    const auto U_i = U_view.template read_tensor<T>(i);

    dealii::Tensor<1, n_comp, T> result;
    result[0] = T(2.) * U_i[0] + U_i[1];
    result[1] = U_i[0] - T(0.5) * U_i[1];

    results_view.template write_tensor<T>(result, i);
  }
};

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

  Vectors::MultiComponentVector<double, n_comp> results;
  results.reinit_with_scalar_partitioner(scalar_partitioner);

  for (unsigned int i = 0; i < n_states; ++i) {
    dealii::Tensor<1, n_comp, double> U_i;
    U_i[0] = 1. + 0.5 * i;
    U_i[1] = -2. + 0.25 * i;
    U.view().write_tensor<double>(U_i, i);
  }

  const auto body = Body{};

  /* Compute results on the host: */

  cpu_simd_loop<double>("loop_01",
                        body,
                        0,
                        /*no vectorization*/ 0,
                        n_states,
                        U.view<HostSpace>(),
                        results.view<HostSpace>());

  std::array<dealii::Tensor<1, n_comp, double>, n_states> host_results;
  for (unsigned int i = 0; i < n_states; ++i)
    host_results[i] = results.view().read_tensor<double>(i);

  /* Reset the results vector and compute the same on the device: */

  for (unsigned int i = 0; i < n_states; ++i)
    results.view().write_tensor<double>(dealii::Tensor<1, n_comp, double>(), i);

  U.move_to_memory_space<DefaultSpace>();
  results.move_to_memory_space<DefaultSpace>();

  gpu_loop<double>("loop_01",
                   body,
                   0,
                   /*ignored*/ 0,
                   n_states,
                   U.view<DefaultSpace>(),
                   results.view<DefaultSpace>());

  results.move_to_memory_space<HostSpace>();

  std::array<dealii::Tensor<1, n_comp, double>, n_states> device_results;
  for (unsigned int i = 0; i < n_states; ++i)
    device_results[i] = results.view().read_tensor<double>(i);

  /* Print all results: */

  std::cout << "Results for " << n_states << " states:\n";
  for (unsigned int i = 0; i < n_states; ++i) {
    std::cout << "\ni = " << i << "\n";
    std::cout << "  host:   " << host_results[i] << "\n";
    std::cout << "  device: " << device_results[i] << "\n";
  }
}
