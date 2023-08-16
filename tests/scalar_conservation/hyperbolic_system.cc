#include <hyperbolic_system.h>
#include <simd.h>

#include <iomanip>
#include <iostream>

using namespace ryujin::ScalarConservation;
using namespace ryujin;
using namespace dealii;


template <int dim, typename Number>
void test(const std::vector<std::string> &flux_description)
{
  std::cout << std::setprecision(10);
  std::cout << std::scientific;

  HyperbolicSystem hyperbolic_system;
  const auto view = hyperbolic_system.view<dim, Number>();

  const std::string expression = std::accumulate(
      std::begin(flux_description),
      std::end(flux_description),
      std::string(),
      [](std::string &result, const std::string &element) {
        return result.empty() ? element : result + "," + element;
      });

  {
    std::stringstream parameters;
    parameters << "subsection HyperbolicSystem\n"
               << "set flux = " << expression << "\n"
               << "end\n"
               << std::endl;
    ParameterAcceptor::initialize(parameters);
  }

  using HyperbolicSystemView = typename HyperbolicSystem::View<dim, Number>;
  using state_type = typename HyperbolicSystemView::state_type;

  state_type U{{1.4}};

  std::cout << "dim = " << dim << std::endl;
  std::cout << "f(u)={" + expression + "}" << std::endl;

  const auto u = view.state(U);
    std::cout << "state = " << u << std::endl;
    std::cout << "flux = " << view.flux_function(u) << std::endl;
    std::cout << "flux_gradient = " << view.flux_gradient_function(u)
              << std::endl;
}

int main()
{
  test<1, double>({"0.5 * u * u"});
  test<1, float>({"0.5 * u * u"});
  test<2, double>({"0.5 * u * u", "0.5 * u * u"});
  test<2, float>({"0.5 * u * u", "0.5 * u * u"});

  test<2, double>({"sin(u)", "cos(u)"});
  test<2, float>({"sin(u)", "cos(u)"});

  return 0;
}
