#include <problem_description.h>

#include <deal.II/base/vectorization.h>

using namespace grendel;
using namespace dealii;


template <int dim, typename Number>
void test()
{
  using rank1_type = typename ProblemDescription<dim, Number>::rank1_type;

  const auto from_1d_state =
      [=](const dealii::Tensor<1, 3, Number> &state_1d) -> rank1_type {
    const auto &rho = state_1d[0];
    const auto &u = state_1d[1];
    const auto &p = state_1d[2];

    rank1_type state;

    state[0] = rho;
    state[1] = rho * u;
    state[dim + 1] =
        p / (ProblemDescription<dim>::gamma - 1.) + 0.5 * rho * u * u;

    return state;
  };

  dealii::Tensor<1, 3, Number> state_1d;
  state_1d[0] = ProblemDescription<dim>::gamma;
  state_1d[1] = 3.;
  state_1d[2] = 1.;
  const auto U = from_1d_state(state_1d);

  std::cout << "dim = " << dim << std::endl;
  std::cout << "momentum = "                                                  //
            << ProblemDescription<dim, Number>::momentum(U)                   //
            << std::endl;                                                     //
  std::cout << "internal_energy = "                                           //
            << ProblemDescription<dim, Number>::internal_energy(U)            //
            << std::endl;                                                     //
  std::cout << "internal_energy_derivative = "                                //
            << ProblemDescription<dim, Number>::internal_energy_derivative(U) //
            << std::endl;                                                     //
  std::cout << "pressure = "                                                  //
            << ProblemDescription<dim, Number>::pressure(U)                   //
            << std::endl;                                                     //
  std::cout << "specific_entropy = "                                          //
            << ProblemDescription<dim, Number>::specific_entropy(U)           //
            << std::endl;                                                     //
  std::cout << "entropy = "                                                   //
            << ProblemDescription<dim, Number>::entropy(U)                    //
            << std::endl;                                                     //
  std::cout << "entropy_derivative = "                                        //
            << ProblemDescription<dim, Number>::entropy_derivative(U)         //
            << std::endl;                                                     //
  std::cout << "f = "                                                         //
            << ProblemDescription<dim, Number>::f(U)                          //
            << std::endl;                                                     //
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
