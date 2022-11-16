
#include <hyperbolic_system.h>
#include <multicomponent_vector.h>
#include <riemann_solver.h>
#define DEBUG_RIEMANN_SOLVER
#include <riemann_solver.template.h>
#include <simd.h>

using namespace ryujin;
using namespace dealii;

int main()
{
  constexpr int dim = 1;

  HyperbolicSystem hyperbolic_system;

  static constexpr unsigned int n_precomputed_values =
      HyperbolicSystem::n_precomputed_values<dim>;
  using precomputed_type = MultiComponentVector<double, n_precomputed_values>;
  precomputed_type dummy;

  RiemannSolver<dim> riemann_solver(hyperbolic_system, dummy);

  // std::stringstream parameters;
  // parameters << "subsection HyperbolicSystem\n"
  //            << "set compute expensive bounds = true\n"
  //            << "end" << std::endl;
  // ParameterAcceptor::initialize(parameters);

  const auto riemann_data = [&](const auto &state) {
    const double rho = state[0];
    const double u = state[1];
    const double p = state[2];
    const double gamma = state[3];

    std::array<double, 5> result;
    result[0] = rho;
    result[1] = u;
    result[2] = p;
    result[3] = gamma;
    const double b_interp = hyperbolic_system.b_interp();
    const double x = 1. - b_interp * rho;
    result[4] = std::sqrt(gamma * p / (rho * x));
    return result;
  };

  const auto test = [&](const std::array<double, 4> &U_i,
                        const std::array<double, 4> &U_j) {
    std::cout << std::endl;
    std::cout << U_i[0] << " " << U_i[1] << " " << U_i[2] << " " << U_i[3]
              << std::endl;
    std::cout << U_j[0] << " " << U_j[1] << " " << U_j[2] << " " << U_j[3]
              << std::endl;
    const auto rd_i = riemann_data(U_i);
    const auto rd_j = riemann_data(U_j);
    const auto lambda_max = riemann_solver.compute(rd_i, rd_j);
    std::cout << lambda_max << std::endl;
  };

  std::cout << std::setprecision(16);
  std::cout << std::scientific;

  /* Test vectors for ideal gas with gamma = 1.4: */

  /* Leblanc:*/
  test({1., 0., 2. / 30., 7. / 5.}, {1.e-3, 0., 2. / 3. * 1.e-10, 7. / 5.});
  /* Sod:*/
  test({1., 0., 1., 7. / 5.}, {0.125, 0., 0.1, 7. / 5.});
  /* Lax:*/
  test({0.445, 0.698, 3.528, 7. / 5.}, {0.5, 0., 0.571, 7. / 5.});
  /* Fast shock case 1 (paper, section 5.2): */
  test({1., 1.e1, 1.e3, 7. / 5.}, {1., 10., 0.01, 7. / 5.});
  /* Fast shock case 2 (paper, section 5.2): */
  test({5.99924, 19.5975, 460.894, 7. / 5.},
       {5.99242, -6.19633, 46.0950, 7. / 5.});
  /* Fast expansion and slow shock, case 1 (Paper, section 5.1) */
  test({1., 0., 0.01, 7. / 5.}, {1., 0., 1.e2, 7. / 5.});
  /* Fast expansion and slow shock, case 2 (Paper, section 5.1) */
  test({1., -1., 0.01, 7. / 5.}, {1., -1., 1.e2, 7. / 5.});
  /* Fast expansion and slow shock, case 3 (Paper, section 5.1) */
  test({1., -2.18, 0.01, 7. / 5.}, {1., -2.18, 100., 7. / 5.});
  /* Case 9:*/
  test({1.0e-2, 0., 1.0e-2, 7. / 5.}, {1.e3, 0., 1.e3, 7. / 5.});
  /* Case 10:*/
  test({1.0, 2.18, 1.e2, 7. / 5.}, {1.0, 2.18, 0.01, 7. / 5.});

  /* States with crazy two-rarefaction pressure: */

  test({1., 300., 1, 1.4}, {0.125, -300., 0.1, 1.4});

  /* States with crazy gamma values: */

  test({1., 0., 2. / 30., 2.99}, {1.e-3, 0., 2. / 3. * 1.e-10, 1.40});

  test({1., 0., 2. / 30., 1.01}, {1.e-3, 0., 2. / 3. * 1.e-10, 1.40});

  test({1., 0., 2. / 30., 2.96}, {1.e-3, 0., 2. / 3. * 1.e-10, 2.99});

  test({1., 0., 2. / 30., 40.0}, {1.e-3, 0., 2. / 3. * 1.e-10, 1.001});

  return 0;
}
