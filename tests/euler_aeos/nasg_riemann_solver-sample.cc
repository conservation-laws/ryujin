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
 * Test NASGRiemannSolverView::sample():
 *
 *  - the ideal gas tests 1 - 5 of Toro, Table 4.1, a case with vacuum
 *    generation, and Leblanc,
 *  - variable gamma, covolume and pinf.
 *
 * We print the solution in every region and verify continuity at the
 * head and tail of rarefaction fans, and the characteristic relation, the
 * isentrope, and the Riemann invariant inside of rarefaction fans.
 * Additionally, we verify that a vectorized sample() with different xi
 * per lane agrees with the scalar variant.
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

  const double l1m = solution.lambda1_minus;
  const double l1p = solution.lambda1_plus;
  const double u_star = solution.u_star;
  const double l3m = solution.lambda3_minus;
  const double l3p = solution.lambda3_plus;

  std::cout << "left:  " << state_i[0] << " " << state_i[1] << " " << state_i[2]
            << " " << state_i[3] << "\n";
  std::cout << "right: " << state_j[0] << " " << state_j[1] << " " << state_j[2]
            << " " << state_j[3] << "\n";

  /*
   * Print the solution at one point in every region:
   */

  const double width = std::max(1., l3p - l1m);
  const std::vector<std::pair<std::string, double>> regions{
      {"left        ", l1m - 0.5 * width},
      {"left fan    ", 0.5 * (l1m + l1p)},
      {"left star   ", 0.5 * (l1p + u_star)},
      {"right star  ", 0.5 * (u_star + l3m)},
      {"right fan   ", 0.5 * (l3m + l3p)},
      {"right       ", l3p + 0.5 * width},
  };

  std::cout << std::setprecision(6);
  for (const auto &[name, xi] : regions) {
    /* Do not sample exactly at a shock: */
    if (xi == l1m || xi == l3p) {
      std::cout << name << "(shock)\n";
      continue;
    }
    const auto [rho, u, p, gamma, a] = solver.sample(solution, xi);
    std::cout << name << "xi = " << std::setw(14) << xi
              << "  [rho, u, p] = " << std::setw(14) << rho << std::setw(14)
              << u << std::setw(14) << p << "\n";
  }
  std::cout << std::setprecision(16);

  const auto check =
      [&](const std::string &name, const double residual, const double scale) {
        if (!(std::abs(residual) <= 1.e-10 * scale))
          std::cout << "FAILED: " << name << " residual = " << residual << "\n";
      };

  const bool vacuum = solution.p_star + pinf <= 0.;

  /*
   * Rarefaction fans:
   */

  std::vector<double> sample_points;
  for (const auto &[name, xi] : regions)
    if (xi != l1m && xi != l3p)
      sample_points.push_back(xi);

  for (const auto &[state, rd, p_star_side, head, tail, sign] :
       {std::tuple{state_i, rd_i, solution.p_star, l1m, l1p, -1.},
        std::tuple{state_j, rd_j, solution.p_star, l3p, l3m, 1.}}) {
    const auto &[rho_Z, u_Z, p_Z, gamma] = state;
    if (p_star_side >= p_Z)
      continue;

    const auto difference = [&](const auto &U, const auto &V, bool front) {
      double result = 0.;
      for (unsigned int k = 0; k < 3; ++k) {
        if (front && k == 1)
          continue; /* the velocity is discontinuous at a vacuum front */
        result = std::max(result,
                          std::abs(U[k] - V[k]) / std::max(1., std::abs(V[k])));
      }
      return result;
    };

    /* Continuity at head and tail of the fan: */
    for (const double xi : {head, tail}) {
      const double delta = 1.e-10 * std::max(1., std::abs(xi));
      const auto U = solver.sample(solution, xi - delta);
      const auto V = solver.sample(solution, xi + delta);
      sample_points.push_back(xi - delta);
      sample_points.push_back(xi + delta);
      const double value = difference(U, V, vacuum && xi == tail);
      if (!(value <= 1.e-7))
        std::cout << "FAILED: continuity at xi = " << xi
                  << " difference = " << value << "\n";
    }

    /* Characteristic, isentrope, and Riemann invariant inside the fan: */
    const double entropy_Z = (p_Z + pinf) * std::pow(1. / rho_Z - b, gamma);
    const double invariant_Z =
        u_Z - sign * 2. * rd[4] * (1. - b * rho_Z) / (gamma - 1.);

    for (const double theta : {0.25, 0.5, 0.75}) {
      const double xi = head + theta * (tail - head);
      const auto [rho, u, p, gamma_xi, a] = solver.sample(solution, xi);

      check("characteristic", xi - (u + sign * a), std::max(1., std::abs(xi)));
      check("isentrope",
            (p + pinf) * std::pow(1. / rho - b, gamma) - entropy_Z,
            entropy_Z);
      check("Riemann invariant",
            u - sign * 2. * a * (1. - b * rho) / (gamma - 1.) - invariant_Z,
            std::max(1., std::abs(invariant_Z)));
    }
  }

  /*
   * Vectorized sample() with a different xi in every lane:
   */

  bool simd_matches = true;
  for (unsigned int n = 0; n < sample_points.size(); n += VA::size()) {
    VA xi;
    for (unsigned int l = 0; l < VA::size(); ++l)
      xi[l] = sample_points[(n + l) % sample_points.size()];

    const auto V = vsolver.sample(vsolution, xi);
    for (unsigned int l = 0; l < VA::size(); ++l) {
      const auto U = solver.sample(solution, xi[l]);
      for (unsigned int k = 0; k < 5; ++k)
        simd_matches &=
            std::abs(V[k][l] - U[k]) <= 1.e-12 * std::max(1., std::abs(U[k]));
    }
  }

  std::cout << (simd_matches ? "" : "SIMD MISMATCH\n") << std::endl;
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

  std::cout << "Ideal gas (Toro, Table 4.1):\n" << std::endl;

  const auto ideal_gas_test = [&](const std::array<double, 3> &U_i,
                                  const std::array<double, 3> &U_j) {
    test(ideal_gas_solver,
         0.,
         0.,
         {U_i[0], U_i[1], U_i[2], gamma},
         {U_j[0], U_j[1], U_j[2], gamma});
  };

  /* Test 1 - 5: */
  ideal_gas_test({1., 0., 1.}, {0.125, 0., 0.1});
  ideal_gas_test({1., -2., 0.4}, {1., 2., 0.4});
  ideal_gas_test({1., 0., 1000.}, {1., 0., 0.01});
  ideal_gas_test({1., 0., 0.01}, {1., 0., 100.});
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
