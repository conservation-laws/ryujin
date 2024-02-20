// force distinct symbols in test
#define ScalarConservation ScalarConservationTest

#include <hyperbolic_system.h>
#include <multicomponent_vector.h>
#define DEBUG_RIEMANN_SOLVER
#include <riemann_solver.h>
#include <riemann_solver.template.h>

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
  typename RiemannSolver<dim, Number>::Parameters riemann_solver_parameters;

  const auto view = hyperbolic_system.view<dim, Number>();

  {
    std::stringstream parameters;
    parameters << "subsection HyperbolicSystem\n"
               << "set flux = " << expression << "\n"
               << "end\n"
               << "subsection RiemannSolver\n"
               << "set use greedy wavespeed = true\n"
               << "set use averaged entropy = true\n"
               << "end\n"
               << std::endl;
    ParameterAcceptor::initialize(parameters);
  }

  using HyperbolicSystemView = typename HyperbolicSystem::View<dim, Number>;
  using state_type = typename HyperbolicSystemView::state_type;
  using precomputed_state_type =
      typename HyperbolicSystemView::precomputed_state_type;
  static constexpr unsigned int n_precomputed_values =
      HyperbolicSystem::View<dim, Number>::n_precomputed_values;
  using precomputed_type = MultiComponentVector<Number, n_precomputed_values>;

  precomputed_type dummy;

  RiemannSolver<dim> riemann_solver(
      hyperbolic_system, riemann_solver_parameters, dummy);

  std::cout << "\n\ndim = " << dim << std::endl;
  std::cout << "f(u)={" + expression + "}" << std::endl;

  Number u_i{1.0};
  const auto f_i = view.flux_function(u_i);
  const auto df_i = view.flux_gradient_function(u_i);

  Number u_j{2.0};
  const auto f_j = view.flux_function(u_j);
  const auto df_j = view.flux_gradient_function(u_j);

  precomputed_state_type prec_i;
  precomputed_state_type prec_j;

  for (unsigned int k = 0; k < n_precomputed_values / 2; ++k) {
    prec_i[k] = f_i[k];
    prec_i[dim + k] = df_i[k];
    prec_j[k] = f_j[k];
    prec_j[dim + k] = df_j[k];
  }

  dealii::Tensor<1, dim, Number> n_ij;

  if constexpr (dim == 1) {
    n_ij[0] = 1.;
    riemann_solver.compute(u_i, u_j, prec_i, prec_j, n_ij);

    n_ij[0] = -1.;
    riemann_solver.compute(u_i, u_j, prec_i, prec_j, n_ij);

  } else if constexpr (dim == 2) {
    n_ij[0] = 1.;
    n_ij[1] = 0.;
    riemann_solver.compute(u_i, u_j, prec_i, prec_j, n_ij);

    n_ij[0] = 0.;
    n_ij[1] = 1.;
    riemann_solver.compute(u_i, u_j, prec_i, prec_j, n_ij);

    n_ij[0] = 1.;
    n_ij[1] = 1.;
    n_ij /= n_ij.norm();
    riemann_solver.compute(u_i, u_j, prec_i, prec_j, n_ij);
  }
}

int main()
{
  test<1, double>("burgers");
  dealii::ParameterAcceptor::clear();
  test<2, double>("burgers");
  dealii::ParameterAcceptor::clear();

  test<2, double>("kpp");

  return 0;
}
