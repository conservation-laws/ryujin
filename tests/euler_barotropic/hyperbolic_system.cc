#include <hyperbolic_system.h>
#include <simd.h>

#include <deal.II/base/vectorization.h>

#include <iomanip>
#include <iostream>

/*
 * Test EOS independent functions:
 */

using namespace ryujin::EulerBarotropic;
using namespace ryujin;
using namespace dealii;


template <int dim, typename Number>
void test(const HyperbolicSystem &hyperbolic_system)
{
  std::cout << std::setprecision(10);
  std::cout << std::scientific;

  const auto view = hyperbolic_system.view<dim, Number>();

  using View = HyperbolicSystemView<dim, Number>;
  using state_type = typename View::state_type;

  const auto from_1d_state =
      [](const dealii::Tensor<1, 2, Number> &state_1d) -> state_type {
    const auto &rho = state_1d[0];
    const auto &u = state_1d[1];

    state_type state;

    state[0] = rho;
    state[1] = rho * u;

    return state;
  };

  dealii::Tensor<1, 2, Number> state_1d;
  state_1d[0] = 1.;
  state_1d[1] = 3.;
  const auto U = from_1d_state(state_1d);

  const Number rho = view.density(U);
  const Number e = view.beos_specific_internal_energy(rho);
  const Number p = view.beos_pressure(rho);
  const Number a = view.beos_speed_of_sound(rho);
  const Number E = view.total_energy(U, e);
  const state_type dE = view.total_energy_derivative(U, e, p);

  std::cout << "dim = " << dim << std::endl;
  std::cout << "density = "                  //
            << view.density(U)               //
            << std::endl;                    //
  std::cout << "momentum = "                 //
            << view.momentum(U)              //
            << std::endl;                    //
  std::cout << "specific internal energy = " //
            << e                             //
            << std::endl;                    //
  std::cout << "total energy = "             //
            << E                             //
            << std::endl;                    //
  std::cout << "total energy derivative = "  //
            << dE                            //
            << std::endl;                    //
  std::cout << "pressure = "                 //
            << p                             //
            << std::endl;                    //
  std::cout << "speed of sound = "           //
            << a                             //
            << std::endl;                    //
  std::cout << "f = "                        //
            << view.f(U, p)                  //
            << std::endl;                    //
}

int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  HyperbolicSystem hyperbolic_system;

  const auto set_eos = [&](const std::string &eos) {
    std::stringstream parameters;
    parameters << "subsection HyperbolicSystem\n"
               << "set barotropic equation of state = " << eos << "\n"
               << "end\n"
               << std::endl;
    ParameterAcceptor::initialize(parameters);
  };

  set_eos("isentropic");
  std::cout << "\ndouble:\n" << std::endl;
  test<1, double>(hyperbolic_system);
  test<2, double>(hyperbolic_system);
  test<3, double>(hyperbolic_system);
  std::cout << "\nfloat:\n" << std::endl;
  test<1, float>(hyperbolic_system);
  test<2, float>(hyperbolic_system);
  test<3, float>(hyperbolic_system);

  set_eos("isothermal");
  std::cout << "\ndouble:\n" << std::endl;
  test<1, double>(hyperbolic_system);
  test<2, double>(hyperbolic_system);
  test<3, double>(hyperbolic_system);
  std::cout << "\nfloat:\n" << std::endl;
  test<1, float>(hyperbolic_system);
  test<2, float>(hyperbolic_system);
  test<3, float>(hyperbolic_system);

  set_eos("function");
  std::cout << "\ndouble:\n" << std::endl;
  test<1, double>(hyperbolic_system);
  test<2, double>(hyperbolic_system);
  test<3, double>(hyperbolic_system);
  std::cout << "\nfloat:\n" << std::endl;
  test<1, float>(hyperbolic_system);
  test<2, float>(hyperbolic_system);
  test<3, float>(hyperbolic_system);

  return 0;
}
