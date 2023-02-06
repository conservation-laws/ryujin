#include <hyperbolic_system.h>
#include <multicomponent_vector.h>
#include <riemann_solver.h>
#include <riemann_solver.template.h>
#include <simd.h>

using namespace ryujin::Euler;
using namespace ryujin;
using namespace dealii;

int main()
{
  constexpr int dim = 1;

  HyperbolicSystem hyperbolic_system;
  const double gamma = hyperbolic_system.gamma();

  static constexpr unsigned int n_precomputed_values =
      HyperbolicSystem::n_precomputed_values<dim>;
  using precomputed_type = MultiComponentVector<double, n_precomputed_values>;
  precomputed_type dummy;

  RiemannSolver<dim> riemann_solver(hyperbolic_system, dummy);

  const auto riemann_data = [&](const auto &state) {
    const double rho = state[0];
    const double u = state[1];
    const double p = state[2];

    std::array<double, 4> result;
    result[0] = rho;
    result[1] = u;
    result[2] = p;
    result[3] = std::sqrt(gamma * p / rho);
    return result;
  };

  const auto test = [&](const std::array<double, 3> &U_i,
                        const std::array<double, 3> &U_j) {
    std::cout << U_i[0] << " " << U_i[1] << " " << U_i[2] << std::endl;
    std::cout << U_j[0] << " " << U_j[1] << " " << U_j[2] << std::endl;
    const auto rd_i = riemann_data(U_i);
    const auto rd_j = riemann_data(U_j);
    const auto lambda_max = riemann_solver.compute(rd_i, rd_j);
    std::cout << lambda_max << std::endl;
  };

  std::cout << std::setprecision(16);
  std::cout << std::scientific;

  std::cout << "gamma: " << gamma << std::endl;
  std::cout << std::endl;

  /* Leblanc:*/
  test({1., 0., 2. / 30.}, {1.e-3, 0., 2. / 3. * 1.e-10});
  /* Sod:*/
  test({1., 0., 1.}, {0.125, 0., 0.1});
  /* Lax:*/
  test({0.445, 0.698, 3.528}, {0.5, 0., 0.571});
  /* Fast shock case 1 (paper, section 5.2): */
  test({1., 1.e1, 1.e3}, {1., 10., 0.01});
  /* Fast shock case 2 (paper, section 5.2): */
  test({5.99924, 19.5975, 460.894}, {5.99242, -6.19633, 46.0950});
  /* Fast expansion and slow shock, case 1 (Paper, section 5.1) */
  test({1., 0., 0.01}, {1., 0., 1.e2});
  /* Fast expansion and slow shock, case 2 (Paper, section 5.1) */
  test({1., -1., 0.01}, {1., -1., 1.e2});
  /* Fast expansion and slow shock, case 3 (Paper, section 5.1) */
  test({1., -2.18, 0.01}, {1., -2.18, 100.});
  /* Case 9:*/
  test({1.0e-2, 0., 1.0e-2}, {1.e3, 0., 1.e3});
  /* Case 10:*/
  test({1.0, 2.18, 1.e2}, {1.0, 2.18, 0.01});

  return 0;
}
