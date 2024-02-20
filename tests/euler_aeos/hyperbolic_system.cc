#include <hyperbolic_system.h>
#include <simd.h>

#include <deal.II/base/vectorization.h>

#include <iomanip>
#include <iostream>

/*
 * Test EOS independent functions:
 */

using namespace ryujin::EulerAEOS;
using namespace ryujin;
using namespace dealii;


static HyperbolicSystem hyperbolic_system;


template <int dim, typename Number>
void test(const Number gamma)
{
  std::cout << std::setprecision(10);
  std::cout << std::scientific;

  std::cout << "interpolatory covolume: "
            << hyperbolic_system.view<dim, Number>().eos_interpolation_b()
            << std::endl;

  const auto view = hyperbolic_system.view<dim, Number>();

  using HyperbolicSystemView = HyperbolicSystemView<dim, Number>;
  using state_type = typename HyperbolicSystemView::state_type;

  const auto from_1d_state =
      [](const dealii::Tensor<1, 3, Number> &state_1d) -> state_type {
    const auto &rho = state_1d[0];
    const auto &u = state_1d[1];
    const auto &e = state_1d[2]; /* specific internal energy */

    state_type state;

    state[0] = rho;
    state[1] = rho * u;
    state[dim + 1] = rho * e + 0.5 * rho * u * u;

    return state;
  };

  dealii::Tensor<1, 3, Number> state_1d;
  state_1d[0] = gamma;
  state_1d[1] = 3.;
  state_1d[2] = 1. / gamma / (gamma - 1.0); /* specific internal energy */
  const auto U = from_1d_state(state_1d);

  std::cout << "dim = " << dim << std::endl;
  std::cout << "density = "                                            //
            << view.density(U)                                         //
            << std::endl;                                              //
  std::cout << "momentum = "                                           //
            << view.momentum(U)                                        //
            << std::endl;                                              //
  std::cout << "total_energy = "                                       //
            << view.total_energy(U)                                    //
            << std::endl;                                              //
  std::cout << "internal_energy = "                                    //
            << view.internal_energy(U)                                 //
            << std::endl;                                              //
  std::cout << "internal_energy_derivative = "                         //
            << view.internal_energy_derivative(U)                      //
            << std::endl;                                              //
  std::cout << "specific_entropy = "                                   //
            << view.surrogate_specific_entropy(U, gamma)               //
            << std::endl;                                              //
  std::cout << "harten entropy = "                                     //
            << view.surrogate_harten_entropy(U, gamma)                 //
            << std::endl;                                              //
  const auto eta = view.surrogate_harten_entropy(U, gamma);            //
  std::cout << "harten_entropy_derivative = "                          //
            << view.surrogate_harten_entropy_derivative(U, eta, gamma) //
            << std::endl;                                              //
  const auto p = view.surrogate_pressure(U, gamma);                    //
  std::cout << "surrogate_pressure = "                                 //
            << view.surrogate_pressure(U, gamma)                       //
            << std::endl;                                              //
  std::cout << "surrogate_gamma = "                                    //
            << view.surrogate_gamma(U, p)                              //
            << std::endl;                                              //
  std::cout << "f = "                                                  //
            << view.f(U, p)                                            //
            << std::endl;                                              //
}

int main()
{
  const auto set_covolume = [&](const double covolume) {
    /*
     * Set the interpolatory covolume by selecting an equation of state
     * with a covolume.
     */
    std::stringstream parameters;
    parameters << "subsection HyperbolicSystem\n"
               << "set equation of state = van der waals\n"
               << "subsection van der waals\n"
               << "set covolume b = " << std::to_string(covolume) << "\n"
               << "end\n"
               << "end\n"
               << std::endl;
    ParameterAcceptor::initialize(parameters);
  };

  std::cout << "\ndouble:\n" << std::endl;
  test<1, double>(/* surrogate gamma */ 1.4);
  test<2, double>(/* surrogate gamma */ 1.4);
  test<3, double>(/* surrogate gamma */ 1.4);
  std::cout << "\nfloat:\n" << std::endl;
  test<1, float>(/* surrogate gamma */ 1.4);
  test<2, float>(/* surrogate gamma */ 1.4);
  test<3, float>(/* surrogate gamma */ 1.4);

  std::cout << "\ndouble:\n" << std::endl;
  test<1, double>(/* surrogate gamma */ 1.9);
  test<2, double>(/* surrogate gamma */ 1.9);
  test<3, double>(/* surrogate gamma */ 1.9);
  std::cout << "\nfloat:\n" << std::endl;
  test<1, float>(/* surrogate gamma */ 1.9);
  test<2, float>(/* surrogate gamma */ 1.9);
  test<3, float>(/* surrogate gamma */ 1.9);


  set_covolume(0.1);
  std::cout << "\ndouble:\n" << std::endl;
  test<1, double>(/* surrogate gamma */ 1.4);
  test<2, double>(/* surrogate gamma */ 1.4);
  test<3, double>(/* surrogate gamma */ 1.4);
  std::cout << "\nfloat:\n" << std::endl;
  test<1, float>(/* surrogate gamma */ 1.4);
  test<2, float>(/* surrogate gamma */ 1.4);
  test<3, float>(/* surrogate gamma */ 1.4);

  std::cout << "\ndouble:\n" << std::endl;
  test<1, double>(/* surrogate gamma */ 1.9);
  test<2, double>(/* surrogate gamma */ 1.9);
  test<3, double>(/* surrogate gamma */ 1.9);
  std::cout << "\nfloat:\n" << std::endl;
  test<1, float>(/* surrogate gamma */ 1.9);
  test<2, float>(/* surrogate gamma */ 1.9);
  test<3, float>(/* surrogate gamma */ 1.9);

  set_covolume(0.5);
  std::cout << "\ndouble:\n" << std::endl;
  test<1, double>(/* surrogate gamma */ 1.4);
  test<2, double>(/* surrogate gamma */ 1.4);
  test<3, double>(/* surrogate gamma */ 1.4);
  std::cout << "\nfloat:\n" << std::endl;
  test<1, float>(/* surrogate gamma */ 1.4);
  test<2, float>(/* surrogate gamma */ 1.4);
  test<3, float>(/* surrogate gamma */ 1.4);

  std::cout << "\ndouble:\n" << std::endl;
  test<1, double>(/* surrogate gamma */ 1.9);
  test<2, double>(/* surrogate gamma */ 1.9);
  test<3, double>(/* surrogate gamma */ 1.9);
  std::cout << "\nfloat:\n" << std::endl;
  test<1, float>(/* surrogate gamma */ 1.9);
  test<2, float>(/* surrogate gamma */ 1.9);
  test<3, float>(/* surrogate gamma */ 1.9);

  return 0;
}
