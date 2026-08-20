#include <hyperbolic_system.h>
#include <multicomponent_vector.h>
#include <simd.h>
#include <wave_speed_estimator.h>

#include <iomanip>
#include <iostream>
#include <sstream>

int main(int argc, char *argv[])
{
  //
  // Test that the WaveSpeedEstimatorView can be used on the device memory
  // space:
  //

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  using HostSpace = dealii::MemorySpace::Host;
  using DefaultSpace = dealii::MemorySpace::Default;

  constexpr int dim = 1;
  constexpr unsigned int n_pairs = 4;

  std::cout << std::setprecision(10);
  std::cout << std::scientific;

  ryujin::Euler::HyperbolicSystem hyperbolic_system;
  ryujin::Euler::WaveSpeedEstimator<double> wave_speed_estimator(
      hyperbolic_system);

  using WSEView = ryujin::Euler::WaveSpeedEstimatorView<dim, double>;
  using primitive_type = typename WSEView::primitive_type;

  const auto gamma = hyperbolic_system.view<dim, double>().gamma();

  /* Assemble Riemann data [rho, u, p, a]: */

  const auto riemann_data =
      [&](const double rho, const double u, const double p) {
        return primitive_type{{rho, u, p, std::sqrt(gamma * p / rho)}};
      };

  std::array<primitive_type, n_pairs> data_i;
  std::array<primitive_type, n_pairs> data_j;

  /* Sod shock tube: */
  data_i[0] = riemann_data(1., 0., 1.);
  data_j[0] = riemann_data(0.125, 0., 0.1);
  /* Two expansion waves: */
  data_i[1] = riemann_data(1., -1., 1.);
  data_j[1] = riemann_data(1., 1., 1.);
  /* Two shocks: */
  data_i[2] = riemann_data(1., 2., 1.);
  data_j[2] = riemann_data(1., -2., 1.);
  /* Stationary contrast: */
  data_i[3] = riemann_data(1., 1., 1.);
  data_j[3] = riemann_data(1., 1., 1.);

  /* Set up locally owned and relevant index sets. */

  dealii::IndexSet locally_owned(n_pairs);
  dealii::IndexSet locally_relevant(n_pairs);
  locally_owned.add_range(0, n_pairs);
  locally_relevant.add_range(0, n_pairs);

  const auto scalar_partitioner =
      std::make_shared<dealii::Utilities::MPI::Partitioner>(
          locally_owned, locally_relevant, MPI_COMM_WORLD);

  ryujin::Vectors::MultiComponentVector<double, 1> results;
  results.reinit_with_scalar_partitioner(scalar_partitioner);

  const auto run = [&](const unsigned int newton_max_iterations) {
    std::stringstream parameters;
    parameters << "subsection WaveSpeedEstimator\n"
               << "set newton max iterations = " << newton_max_iterations
               << "\nend" << std::endl;
    dealii::ParameterAcceptor::initialize(parameters);

    /* The views have to be created after parsing parameters: */

    const auto host_view = wave_speed_estimator.view<dim, double, HostSpace>();
    const auto device_view =
        wave_speed_estimator.view<dim, double, DefaultSpace>();

    std::cout << "\nnewton max iterations = " << newton_max_iterations << "\n";

    /* Compute lambda_max on the host: */

    std::cout << "\nComputed on the host:\n";
    for (unsigned int k = 0; k < n_pairs; ++k)
      std::cout << host_view.compute(data_i[k], data_j[k]) << "\n";

    /* Compute lambda_max on the default space: */

    results.move_to_memory_space<DefaultSpace>();
    const auto results_view = results.get_view<DefaultSpace>();

    using ExecutionSpace = DefaultSpace::kokkos_space::execution_space;
    const auto exec = ExecutionSpace{};
    Kokkos::parallel_for("test",
                         Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n_pairs),
                         [=](std::size_t k) {
                           dealii::Tensor<1, 1, double> result;
                           result[0] =
                               device_view.compute(data_i[k], data_j[k]);
                           results_view.write_tensor(result, k);
                         });

    results.move_to_memory_space<HostSpace>();

    std::cout << "\nComputed on the device:\n";
    for (unsigned int k = 0; k < n_pairs; ++k)
      std::cout << results.read_tensor<double>(k)[0] << "\n";
  };

  run(0);
  run(2);
}
