#include <problem_description.h>
#include <simd.template.h>

#include <deal.II/base/vectorization.h>

#include <iostream>
#include <iomanip>

using namespace ryujin;
using namespace dealii;


template <int dim, typename Number>
void test()
{
  std::cout << std::setprecision(10);

  ProblemDescription problem_description;
  problem_description.parse_parameters_callback();

  using state_type = typename ProblemDescription::state_type<dim, Number>;

  const auto from_1d_state =
      [&problem_description](
          const dealii::Tensor<1, 3, Number> &state_1d) -> state_type {
    const auto &rho = state_1d[0];
    const auto &u = state_1d[1];
    const auto &p = state_1d[2];

    state_type state;

    state[0] = rho;
    state[1] = rho * u;
    state[dim + 1] = p / (problem_description.gamma() - 1.) + 0.5 * rho * u * u;

    return state;
  };

  dealii::Tensor<1, 3, Number> state_1d;
  state_1d[0] = problem_description.gamma();
  state_1d[1] = 3.;
  state_1d[2] = 1.;
  const auto U = from_1d_state(state_1d);

  std::cout << "dim = " << dim << std::endl;
  std::cout << "momentum = "                                          //
            << problem_description.momentum(U)                        //
            << std::endl;                                             //
  std::cout << "internal_energy = "                                   //
            << problem_description.internal_energy(U)                 //
            << std::endl;                                             //
  std::cout << "internal_energy_derivative = "                        //
            << problem_description.internal_energy_derivative(U)      //
            << std::endl;                                             //
  std::cout << "pressure = "                                          //
            << problem_description.pressure(U)                        //
            << std::endl;                                             //
  std::cout << "specific_entropy = "                                  //
            << problem_description.specific_entropy(U)                //
            << std::endl;                                             //
  std::cout << "harten entropy = "                                    //
            << problem_description.harten_entropy(U)                  //
            << std::endl;                                             //
  std::cout << "harten_entropy_derivative = "                         //
            << problem_description.harten_entropy_derivative(U)       //
            << std::endl;                                             //
  std::cout << "mathematical entropy = "                              //
            << problem_description.mathematical_entropy(U)            //
            << std::endl;                                             //
  std::cout << "mathematical_entropy_derivative = "                   //
            << problem_description.mathematical_entropy_derivative(U) //
            << std::endl;                                             //
  std::cout << "f = "                                                 //
            << problem_description.f(U)                               //
            << std::endl;                                             //
}

int main()
{
  std::cout << "\ndouble:\n" << std::endl;
  test<1, double>();
  test<2, double>();
  test<3, double>();

  std::cout << "float:\n" << std::endl;
  test<1, float>();
  test<2, float>();
  test<3, float>();

  std::cout << "\nVectorizedArray<double>\n" << std::endl;
  test<1, VectorizedArray<double>>();
  test<2, VectorizedArray<double>>();
  test<3, VectorizedArray<double>>();

  std::cout << "\nVectorizedArray<float>\n" << std::endl;
  test<1, VectorizedArray<float>>();
  test<2, VectorizedArray<float>>();
  test<3, VectorizedArray<float>>();

  return 0;
}
