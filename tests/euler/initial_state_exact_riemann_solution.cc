// force distinct symbols in test
#define Euler EulerTest

#include <hyperbolic_system.h>
#include <multicomponent_vector.h>
#include <simd.h>

#define DEBUG_SOLUTION
#include <description.h>
#include <initial_state_exact_riemann_solution.h>

using namespace ryujin::EulerInitialStates;
using namespace ryujin::Euler;
using namespace ryujin;
using namespace dealii;

int main()
{
  constexpr int dim = 1;

  HyperbolicSystem hyperbolic_system;

  using state_type = HyperbolicSystemView<dim, double>::state_type;

  std::cout << std::setprecision(16);
  std::cout << std::scientific;

  std::cout << "Calling default constructor..." << std::endl;
  ExactRiemannSolution<Description, dim, double> initial_state(
      hyperbolic_system, "");

  {
    std::cout << "Resetting parameters..." << std::endl;
    std::stringstream parameters;
    parameters << "subsection exact riemann solution\n"
               << "set primitive state left =     1., 0., 0.666666666667e-1\n"
               << "set primitive state right = 1.e-3, 0., 0.666666666667e-10\n"
               << "end\n"
               << std::endl;
    ParameterAcceptor::initialize(parameters);
  }

  std::cout << "Calling compute..." << std::endl;
  const auto test = [&](const double x, const double t) {
    std::cout << "Position x = " << x << ", t = " << t << std::endl;
    auto state = initial_state.compute(Point<dim>{x}, t);
    std::cout << "  --> " << state << std::endl;
  };

  for (double t : {0.0, 0.5, 1.0, 1.5, 2.0, 2.5}) {
    test(-0.5, t);
    test(0.5, t);
  }

  return 0;
}
