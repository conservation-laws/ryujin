// force distinct symbols in test
#define EulerAEOS EulerAEOSTest

#include <nasg_riemann_solver.h>
#include <simd.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/vectorization.h>

#include <iomanip>
#include <iostream>
#include <sstream>

using namespace ryujin::EulerAEOS;
using namespace ryujin;
using namespace dealii;

/*
 * Test the NASGRiemannSolver with all compile time options for the ideal
 * gas case (no covolume, no pinf, single gamma, plain divisions) on the
 * test vectors of tests/euler/wave_speed_estimator.cc.
 */

constexpr NASGRiemannSolverOptions ideal_gas{.covolume = false,
                                             .pinf = false,
                                             .safe_division = false,
                                             .variable_gamma = false};

int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  using VA = VectorizedArray<double>;

  constexpr double gamma = 7. / 5.;

  /* One solver without and one with 10 Newton iterations: */
  NASGRiemannSolver<double, ideal_gas> solver_0("/RiemannSolver0");
  NASGRiemannSolver<double, ideal_gas> solver_10("/RiemannSolver10");
  solver_0.set_gamma(gamma);
  solver_10.set_gamma(gamma);

  std::stringstream parameters;
  parameters << "subsection RiemannSolver10\n"
             << "set newton max iterations = 10\n"
             << "end" << std::endl;
  ParameterAcceptor::initialize(parameters);

  const auto riemann_data = [&](const std::array<double, 3> &state) {
    const double rho = state[0];
    const double u = state[1];
    const double p = state[2];
    return std::array<double, 5>{rho, u, p, gamma, std::sqrt(gamma * p / rho)};
  };

  const auto test = [&](const std::array<double, 3> &U_i,
                        const std::array<double, 3> &U_j) {
    std::cout << U_i[0] << " " << U_i[1] << " " << U_i[2] << std::endl;
    std::cout << U_j[0] << " " << U_j[1] << " " << U_j[2] << std::endl;

    const auto rd_i = riemann_data(U_i);
    const auto rd_j = riemann_data(U_j);

    std::array<VA, 5> vrd_i, vrd_j;
    for (unsigned int k = 0; k < 5; ++k) {
      vrd_i[k] = rd_i[k];
      vrd_j[k] = rd_j[k];
    }

    for (const auto *riemann_solver : {&solver_0, &solver_10}) {
      const auto solver = riemann_solver->view<double>();
      const auto vsolver = riemann_solver->view<VA>();
      const unsigned int iterations = solver.newton_max_iterations();

      const auto lambda_max = solver.compute(rd_i, rd_j);
      const auto vlambda_max = vsolver.compute(vrd_i, vrd_j);

      bool simd_matches = true;
      for (unsigned int l = 0; l < VA::size(); ++l)
        simd_matches &=
            std::abs(vlambda_max[l] - lambda_max) <= 1.e-14 * lambda_max;

      std::cout << "iterations " << iterations
                << ": lambda_max = " << lambda_max
                << (simd_matches ? "" : " (SIMD MISMATCH)") << std::endl;
    }
    std::cout << std::endl;
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
