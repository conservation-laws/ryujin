#include <problem_description.h>
#include <riemann_solver.h>

using namespace grendel;
using namespace dealii;

int main()
{
  ProblemDescription<2> problem_description;

  std::cout << std::setprecision(16);
  std::cout << std::scientific;

  ParameterAcceptor::prm.enter_subsection("ProblemDescription");
  const double &gamma = problem_description.gamma();
  const double &b = problem_description.b();
  const auto &state_L = problem_description.initial_state_L();
  const auto &state_R = problem_description.initial_state_R();

  ParameterAcceptor::prm.set("gamma", "1.4");
  ParameterAcceptor::prm.set("b", "0.");
  ParameterAcceptor::prm.set("initial state", "contrast");
  ParameterAcceptor::prm.set("initial - direction", "1., 0.");
  problem_description.parse_parameters_callback();

  std::cout << gamma << std::endl;
  std::cout << b << std::endl;
  std::cout << state_L << std::endl;
  std::cout << state_R << std::endl << std::endl;

  return 0;
}
