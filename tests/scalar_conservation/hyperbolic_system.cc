#include <hyperbolic_system.h>
#include <simd.h>

#include <iomanip>
#include <iostream>

using namespace ryujin::ScalarConservation;
using namespace ryujin;
using namespace dealii;


template <int dim, typename Number>
void test(const std::string &expression)
{
  std::cout << std::setprecision(10);
  std::cout << std::scientific;

  HyperbolicSystem hyperbolic_system;
  const auto view = hyperbolic_system.view<dim, Number>();

  {
    std::stringstream parameters;
    parameters << "subsection HyperbolicSystem\n"
               << "set flux = " << expression << "\n"
               << "end\n"
               << std::endl;
    ParameterAcceptor::initialize(parameters);
  }

  using HyperbolicSystemView = HyperbolicSystemView<dim, Number>;
  using state_type = typename HyperbolicSystemView::state_type;

  state_type U{{1.4}};

  std::cout << "dim = " << dim << std::endl;
  std::cout << "f(u)={" + expression + "}" << std::endl;

  const auto u = view.state(U);
  std::cout << "state = " << u << std::endl;
  std::cout << "square_entropy = " << view.square_entropy(u) << std::endl;
  std::cout << "square_entropy_derivative = "
            << view.square_entropy_derivative(u) << std::endl;
  std::cout << "flux = " << view.flux_function(u) << std::endl;
  std::cout << "flux_gradient = " << view.flux_gradient_function(u)
            << std::endl;
}

int main()
{
  test<1, double>("burgers");
  dealii::ParameterAcceptor::clear();
  test<1, float>("burgers");
  dealii::ParameterAcceptor::clear();
  test<2, double>("burgers");
  dealii::ParameterAcceptor::clear();
  test<2, float>("burgers");
  dealii::ParameterAcceptor::clear();

  test<2, double>("kpp");
  dealii::ParameterAcceptor::clear();
  test<2, float>({"kpp"});

  return 0;
}
