// force distinct symbols in test
#define EulerAEOS EulerAEOSTest

#include <nasg_riemann_solver.h>
#include <simd.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/vectorization.h>

#include <iomanip>
#include <iostream>

using namespace ryujin::EulerAEOS;
using namespace ryujin;
using namespace dealii;

/*
 * Test NASGRiemannSolverView::solve():
 *
 *  - the ideal gas tests 1 - 5 of Toro, Table 4.1 (exact values in
 *    Table 4.3), and a case with vacuum generation,
 *  - variable gamma, covolume and pinf, where we verify phi(p_star) = 0,
 *    mass conservation across shocks, and the isentrope across
 *    rarefaction waves.
 */

constexpr NASGRiemannSolverOptions ideal_gas{.covolume = false,
                                             .pinf = false,
                                             .safe_division = true,
                                             .variable_gamma = false};

constexpr NASGRiemannSolverOptions nasg{};

template <typename Solver>
void test(const Solver &riemann_solver,
          const double b,
          const double pinf,
          const std::array<double, 4> &state_i,
          const std::array<double, 4> &state_j)
{
  using VA = VectorizedArray<double>;

  /* state = [rho, u, p, gamma] */
  const auto riemann_data = [&](const std::array<double, 4> &state) {
    const auto &[rho, u, p, gamma] = state;
    const double a = std::sqrt(gamma * (p + pinf) / (rho * (1. - b * rho)));
    return std::array<double, 5>{rho, u, p, gamma, a};
  };

  const auto rd_i = riemann_data(state_i);
  const auto rd_j = riemann_data(state_j);

  std::array<VA, 5> vrd_i, vrd_j;
  for (unsigned int k = 0; k < 5; ++k) {
    vrd_i[k] = rd_i[k];
    vrd_j[k] = rd_j[k];
  }

  const auto solver = riemann_solver.template view<double>();
  const auto vsolver = riemann_solver.template view<VA>();

  const auto solution = solver.solve(rd_i, rd_j);
  const auto vsolution = vsolver.solve(vrd_i, vrd_j);

  const std::array<double, 8> values{solution.p_star,
                                     solution.u_star,
                                     solution.rho_star_left,
                                     solution.rho_star_right,
                                     solution.lambda1_minus,
                                     solution.lambda1_plus,
                                     solution.lambda3_minus,
                                     solution.lambda3_plus};
  const std::array<VA, 8> vvalues{vsolution.p_star,
                                  vsolution.u_star,
                                  vsolution.rho_star_left,
                                  vsolution.rho_star_right,
                                  vsolution.lambda1_minus,
                                  vsolution.lambda1_plus,
                                  vsolution.lambda3_minus,
                                  vsolution.lambda3_plus};

  bool simd_matches = true;
  for (unsigned int k = 0; k < values.size(); ++k)
    for (unsigned int l = 0; l < VA::size(); ++l)
      simd_matches &= std::abs(vvalues[k][l] - values[k]) <=
                      1.e-14 * std::max(1., std::abs(values[k]));

  const auto &[riemann_data_left,
               riemann_data_right,
               p_star,
               u_star,
               rho_star_left,
               rho_star_right,
               lambda1_minus,
               lambda1_plus,
               lambda3_minus,
               lambda3_plus] = solution;

  std::cout << "left:  " << state_i[0] << " " << state_i[1] << " " << state_i[2]
            << " " << state_i[3] << "\n";
  std::cout << "right: " << state_j[0] << " " << state_j[1] << " " << state_j[2]
            << " " << state_j[3] << "\n";

  std::cout << std::setprecision(5);
  std::cout << "p_star         = " << p_star << "\n"
            << "u_star         = " << u_star << "\n"
            << "rho_star_left  = " << rho_star_left << "\n"
            << "rho_star_right = " << rho_star_right << "\n"
            << "lambda1_minus  = " << lambda1_minus << "\n"
            << "lambda1_plus   = " << lambda1_plus << "\n"
            << "lambda3_minus  = " << lambda3_minus << "\n"
            << "lambda3_plus   = " << lambda3_plus << "\n";
  std::cout << std::setprecision(16);

  const bool ordered = lambda1_minus <= lambda1_plus &&
                       lambda1_plus <= u_star && u_star <= lambda3_minus &&
                       lambda3_minus <= lambda3_plus;

  /*
   * Consistency checks. Relative residuals are compared against a
   * tolerance:
   */

  const auto check =
      [&](const std::string &name, const double residual, const double scale) {
        if (std::abs(residual) > 1.e-10 * scale)
          std::cout << "FAILED: " << name << " residual = " << residual << "\n";
      };

  const bool vacuum = p_star + pinf <= 0.;

  if (!vacuum) {
    const double phi = solver.phi(rd_i, rd_j, p_star);
    check("phi(p_star)",
          phi,
          std::max({1., std::abs(state_i[1]), std::abs(state_j[1]), rd_i[4]}));
  }

  for (const auto &[state, rho_star, lambda] :
       {std::tuple{state_i, rho_star_left, lambda1_minus},
        std::tuple{state_j, rho_star_right, lambda3_plus}}) {
    const auto &[rho, u, p, gamma] = state;
    if (vacuum) {
      check("rho_star (vacuum)", rho_star, 1.);
    } else if (p_star >= p) {
      /* Mass conservation across the shock: */
      check("Rankine-Hugoniot",
            rho * (u - lambda) - rho_star * (u_star - lambda),
            rho * std::abs(u - lambda));
    } else {
      /* Isentrope (p + pinf) (1 / rho - b)^gamma = const: */
      const double left = (p + pinf) * std::pow(1. / rho - b, gamma);
      const double right = (p_star + pinf) * std::pow(1. / rho_star - b, gamma);
      check("isentrope", left - right, left);
    }
  }

  std::cout << (ordered ? "" : "WAVE SPEEDS NOT ORDERED\n")
            << (simd_matches ? "" : "SIMD MISMATCH\n") << std::endl;
}


int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  constexpr double gamma = 7. / 5.;

  NASGRiemannSolver<double, ideal_gas> ideal_gas_solver("/IdealGas");
  ideal_gas_solver.set_gamma(gamma);

  NASGRiemannSolver<double, nasg> nasg_solver("/NASG");

  ParameterAcceptor::initialize();

  std::cout << std::setprecision(16);
  std::cout << std::scientific;

  std::cout << "Ideal gas (Toro, Table 4.1 and 4.3):\n" << std::endl;

  const auto ideal_gas_test = [&](const std::array<double, 3> &U_i,
                                  const std::array<double, 3> &U_j) {
    test(ideal_gas_solver,
         0.,
         0.,
         {U_i[0], U_i[1], U_i[2], gamma},
         {U_j[0], U_j[1], U_j[2], gamma});
  };

  /* Test 1: p* = 0.30313, u* = 0.92745, rho*L = 0.42632, rho*R = 0.26557 */
  ideal_gas_test({1., 0., 1.}, {0.125, 0., 0.1});
  /* Test 2: p* = 0.00189, u* = 0., rho*L = 0.02185, rho*R = 0.02185 */
  ideal_gas_test({1., -2., 0.4}, {1., 2., 0.4});
  /* Test 3: p* = 460.894, u* = 19.5975, rho*L = 0.57506, rho*R = 5.99924 */
  ideal_gas_test({1., 0., 1000.}, {1., 0., 0.01});
  /* Test 4: p* = 46.0950, u* = -6.19633, rho*L = 5.99242, rho*R = 0.57511 */
  ideal_gas_test({1., 0., 0.01}, {1., 0., 100.});
  /* Test 5: p* = 1691.64, u* = 8.68975, rho*L = 14.2823, rho*R = 31.0426 */
  ideal_gas_test({5.99924, 19.5975, 460.894}, {5.99242, -6.19633, 46.0950});
  /* Vacuum generation: */
  ideal_gas_test({1., -4., 0.4}, {1., 4., 0.4});
  /* Leblanc: */
  ideal_gas_test({1., 0., 2. / 30.}, {1.e-3, 0., 2. / 3. * 1.e-10});

  std::cout << "NASG, variable gamma:\n" << std::endl;

  for (const auto &[b, pinf] : {std::pair{0., 0.},
                                std::pair{0.1, 0.},
                                std::pair{0., 2.},
                                std::pair{0.1, 2.}}) {
    std::cout << "b = " << b << ", pinf = " << pinf << "\n" << std::endl;
    nasg_solver.set_equation_of_state(b, pinf, false);

    /* rarefaction - shock */
    test(nasg_solver, b, pinf, {1., 0., 1., 1.4}, {0.125, 0., 0.1, 5. / 3.});
    /* shock - shock */
    test(nasg_solver, b, pinf, {1., 2., 1., 1.4}, {2., -2., 1., 3.});
    /* rarefaction - rarefaction */
    test(nasg_solver, b, pinf, {1., -1., 1., 1.4}, {1.5, 1., 2., 2.});
    /* vacuum generation */
    test(nasg_solver, b, pinf, {1., -20., 1., 1.4}, {1., 20., 1., 1.4});
  }

  return 0;
}
