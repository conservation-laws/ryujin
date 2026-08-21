// force distinct symbols in test
#define EulerBarotropic EulerBarotropicTest

#include <hyperbolic_system.h>
#include <multicomponent_vector.h>
#define DEBUG_WAVE_SPEED_ESTIMATOR
#include <simd.h>
#include <wave_speed_estimator.h>
#include <wave_speed_estimator.template.h>

using namespace ryujin::EulerBarotropic;
using namespace ryujin;
using namespace dealii;

int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  constexpr int dim = 1;

  HyperbolicSystem hyperbolic_system;
  WaveSpeedEstimator<double> wave_speed_estimator(hyperbolic_system);

  const auto wave_speed_estimator_view =
      wave_speed_estimator.view<dim, double>();

  const auto view = hyperbolic_system.view<dim, double>();

  using state_type = dealii::Tensor<1, 1 + dim, double>;

  const auto riemann_data = [&](const state_type &state) {
    const double rho = view.density(state);
    const double m = view.momentum(state)[0];
    const double u = m / rho;
    const double a = view.beos_speed_of_sound(rho);

    std::array<double, 2> result;
    result[0] = u;
    result[1] = a;

    return result;
  };

  const auto test = [&](const state_type &U_i, const state_type &U_j) {
    std::cout << std::endl;
    std::cout << U_i[0] << " " << U_i[1] << std::endl;
    std::cout << U_j[0] << " " << U_j[1] << std::endl;
    const auto rd_i = riemann_data(U_i);
    const auto rd_j = riemann_data(U_j);
    const auto lambda_max = wave_speed_estimator_view.compute(rd_i, rd_j);
    std::cout << lambda_max << std::endl;
  };

  const auto set_eos = [&](const std::string &eos) {
    std::stringstream parameters;
    parameters << "subsection HyperbolicSystem\n"
               << "set barotropic equation of state = " << eos << "\n"
               << "end\n"
               << std::endl;
    ParameterAcceptor::initialize(parameters);
  };

  std::cout << std::setprecision(16);
  std::cout << std::scientific;

  // TODO: implement actual test cases

  state_type U_i;
  U_i[0] = 1.;
  U_i[1] = 0.;

  state_type U_j;
  U_j[0] = 1.;
  U_j[1] = 0.;

  set_eos("isentropic");
  test(U_i, U_j);

  set_eos("isothermal");
  test(U_i, U_j);

  set_eos("function");
  test(U_i, U_j);

  return 0;
}
