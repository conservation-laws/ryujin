// force distinct symbols in test
#define ShallowWater ShallowWaterTest

#include <hyperbolic_system.h>
#include <multicomponent_vector.h>
#define DEBUG_RIEMANN_SOLVER
#include <riemann_solver.h>
#include <riemann_solver.template.h>
#include <simd.h>

using namespace ryujin::ShallowWater;
using namespace ryujin;
using namespace dealii;

int main()
{
  constexpr int dim = 1;

  HyperbolicSystem hyperbolic_system;
  const double gravity = hyperbolic_system.view<dim, double>().gravity();

  RiemannSolver<dim, double>::Parameters riemann_solver_parameters;

  static constexpr unsigned int n_precomputed_values =
      HyperbolicSystemView<dim, double>::n_precomputed_values;
  using precomputed_type = MultiComponentVector<double, n_precomputed_values>;
  precomputed_type dummy;

  RiemannSolver<dim> riemann_solver(
      hyperbolic_system, riemann_solver_parameters, dummy);

  const auto riemann_data = [&](const auto &state) {
    const auto h = hyperbolic_system.view<dim, double>().water_depth_sharp(
        dealii::Tensor<1, 2, double>{{state[0], state[1]}});
    const double u = state[1] / h;

    std::array<double, 3> result;
    result[0] = h;
    result[1] = u;
    result[2] = std::sqrt(gravity * h);
    return result;
  };

  const auto test = [&](const std::array<double, 2> &U_i,
                        const std::array<double, 2> &U_j) {
    std::cout << U_i[0] << " " << U_i[1] << std::endl;
    std::cout << U_j[0] << " " << U_j[1] << std::endl;
    const auto rd_i = riemann_data(U_i);
    const auto rd_j = riemann_data(U_j);
    std::cout << "relaxation: " << rd_i[0] << " " << rd_i[1] << std::endl;
    std::cout << "relaxation: " << rd_j[0] << " " << rd_j[1] << std::endl;
    const auto h_star = riemann_solver.compute_h_star(rd_i, rd_j);
    const auto lambda_max = riemann_solver.compute(rd_i, rd_j);
    std::cout << "lambda_max: " << lambda_max << std::endl;
    std::cout << "h_star: " << h_star << std::endl;
    std::cout << std::endl;
  };

  std::cout << std::setprecision(16);
  std::cout << std::scientific;

  const auto view = hyperbolic_system.view<dim, double>();

  std::cout << "gravity:                      " << gravity << std::endl;
  std::cout << "reference_water_depth:        " << view.reference_water_depth()
            << std::endl;
  std::cout << "dry_state_relaxation (small): "
            << view.dry_state_relaxation_small() << std::endl;
  std::cout << "dry_state_relaxation (large): "
            << view.dry_state_relaxation_large() << std::endl;
  std::cout << std::endl;

  // 10/04/2022 verified against Mathematica computation (Eric + Matthias)
  test({0.0, 0.0}, {0.0, 0.0});
  test({1.0, 1.0}, {0.0, 0.0});
  test({1.8, 0.0}, {1.0, 0.0});

  return 0;
}
