#include <problem_description.h>
#include <riemann_solver.h>
#include <riemann_solver.template.h>

using namespace grendel;
using namespace dealii;

int main()
{
  constexpr int dim = 2;

  const auto riemann_data = [&](const auto &state) {
    const double rho = state[0];
    const double u = state[1];
    const double p = state[2];

    std::array<double, 4> result;
    result[0] = rho;
    result[1] = u;
    result[2] = p;
    constexpr double gamma = ProblemDescription<dim>::gamma;
    result[3] = std::sqrt(gamma * p / rho);
    return result;
  };

  const auto test = [&](const std::array<double, 3> &U_i,
                        const std::array<double, 3> &U_j) {
    std::cout << U_i[0] << " " << U_i[1] << " " << U_i[2] << std::endl;
    std::cout << U_j[0] << " " << U_j[1] << " " << U_j[2] << std::endl;
    const auto rd_i = riemann_data(U_i);
    const auto rd_j = riemann_data(U_j);
    const auto [lambda_max, p_star, n_iterations] =
        RiemannSolver<dim>::compute</* max_iter = */ 10>(rd_i, rd_j);
    std::cout << lambda_max << std::endl;
    std::cout << p_star << std::endl;
    std::cout << n_iterations << std::endl << std::endl;
  };

  const auto test_quick = [&](const std::array<double, 3> &U_i,
                              const std::array<double, 3> &U_j) {
    std::cout << U_i[0] << " " << U_i[1] << " " << U_i[2] << std::endl;
    std::cout << U_j[0] << " " << U_j[1] << " " << U_j[2] << std::endl;
    const auto rd_i = riemann_data(U_i);
    const auto rd_j = riemann_data(U_j);
    const auto [lambda_max, p_star, n_iterations] =
        RiemannSolver<dim>::compute</* max_iter = */ 0>(rd_i, rd_j);
    std::cout << lambda_max << std::endl;
    std::cout << p_star << std::endl;
    std::cout << n_iterations << std::endl << std::endl;
  };

  std::cout << std::setprecision(16);
  std::cout << std::scientific;

  constexpr double gamma = ProblemDescription<dim>::gamma;
  constexpr double b = ProblemDescription<dim>::b;


  std::cout << "gamma: " << gamma << std::endl;
  std::cout << "b: " << b << std::endl;
  std::cout << "max_iter:  " << 10 << std::endl;
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

  std::cout << "gamma: " << gamma << std::endl;
  std::cout << "b: " << b << std::endl;
  std::cout << "max_iter:  " << 0 << std::endl;
  std::cout << std::endl;

  /* Leblanc:*/
  test_quick({1., 0., 2. / 30.}, {1.e-3, 0., 2. / 3. * 1.e-10});
  /* Sod:*/
  test_quick({1., 0., 1.}, {0.125, 0., 0.1});
  /* Lax:*/
  test_quick({0.445, 0.698, 3.528}, {0.5, 0., 0.571});
  /* Fast shock case 1 (paper, section 5.2): */
  test_quick({1., 1.e1, 1.e3}, {1., 10., 0.01});
  /* Fast shock case 2 (paper, section 5.2): */
  test_quick({5.99924, 19.5975, 460.894}, {5.99242, -6.19633, 46.0950});
  /* Fast expansion and slow shock, case 1 (Paper, section 5.1) */
  test_quick({1., 0., 0.01}, {1., 0., 1.e2});
  /* Fast expansion and slow shock, case 2 (Paper, section 5.1) */
  test_quick({1., -1., 0.01}, {1., -1., 1.e2});
  /* Fast expansion and slow shock, case 3 (Paper, section 5.1) */
  test_quick({1., -2.18, 0.01}, {1., -2.18, 100.});
  /* Case 9:*/
  test_quick({1.0e-2, 0., 1.0e-2}, {1.e3, 0., 1.e3});
  /* Case 10:*/
  test_quick({1.0, 2.18, 1.e2}, {1.0, 2.18, 0.01});

  return 0;
}
