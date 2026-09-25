//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by Jerett Cherry
//

//
// Regression test for the compute_pstar() fix in
// initial_state_exact_riemann_solution.h: compute_pstar() used to enforce
// p_1 <= p_2 by swapping *both* the two pressures and the two associated
// left/right state tensors. That swap is not a symmetry of phi -- the "f"
// sum is symmetric in left/right, but the u_R - u_L term of phi,
// @cite GuermondPopov2016b, page 912, (3.3), is antisymmetric -- so
// transposing the states shifts phi by the constant -2(u_R - u_L) and
// moves the root whenever u_L != u_R. The fix orders the two pressures as
// scalars only (Algorithm 1, line 1) and never transposes the states.
//
// This test checks the fix the direct way that matters physically: the
// maximal wave speed lambda_max implied by the *exact* self-similar
// solution (recovered here purely through ExactRiemannSolution::compute(),
// its only public interface) must agree with the *approximate* lambda_max
// returned by WaveSpeedEstimator -- which never had this defect, since it
// only ever orders p_min/p_max as scalars and always evaluates phi with
// the states in their original left/right slots. Before the fix, the two
// disagreed for any u_L != u_R; the case picked here is exactly such a
// case.
//
// Test data: Toro, "Riemann Solvers and Numerical Methods for Fluid
// Dynamics", 3rd ed., Chapter 4, Test 1 (the "modified Sod" problem, so
// named because u_L = 0.75 != 0 = u_R) -- the same configuration as
// prm/verification/euler-toro_1-erk33.prm.
//

// force distinct symbols in test
#define Euler EulerTest

#include <hyperbolic_system.h>
#include <simd.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>

#include <description.h>
#include <initial_state_exact_riemann_solution.h>
#include <wave_speed_estimator.h>

using namespace ryujin::EulerInitialStates;
using namespace ryujin::Euler;
using namespace ryujin;
using namespace dealii;

namespace
{
  /**
   * ExactRiemannSolution keeps the two outer characteristic speeds of the
   * self-similar solution (Guermond & Popov's lambda^-_1 and lambda^+_3,
   * i.e., the speeds bounding the whole Riemann fan) private. We recover
   * them here using nothing but the public compute() interface: for
   * xi = x/t on the far side of one of these speeds, compute() returns
   * the untouched far-field state @p far bit-for-bit; immediately past
   * it, it does not (it enters the fan). So the transition point of the
   * predicate "compute(x, 1) == far" *is* the wave speed, and we find it
   * by bisection. @p lo and @p hi must bracket the transition, i.e., the
   * predicate must differ at the two endpoints; which one starts "true"
   * does not matter.
   */
  template <int dim, typename ExactSolution, typename StateType>
  double bisect_wave_speed(ExactSolution &exact,
                           const StateType &far,
                           double lo,
                           double hi)
  {
    const bool eq_lo = (exact.compute(Point<dim>{lo}, 1.) == far);
    AssertThrow(eq_lo != (exact.compute(Point<dim>{hi}, 1.) == far),
                ExcMessage("bisect_wave_speed: [lo, hi] does not bracket a "
                           "single sign change"));

    for (unsigned int i = 0; i < 100; ++i) {
      const double mid = 0.5 * (lo + hi);
      const bool eq_mid = (exact.compute(Point<dim>{mid}, 1.) == far);
      if (eq_mid == eq_lo)
        lo = mid;
      else
        hi = mid;
    }
    return 0.5 * (lo + hi);
  }
} // namespace


int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  constexpr int dim = 1;

  /* Toro, Chapter 4, Test 1 ("modified Sod"): u_L != u_R. */
  const double rho_L = 1., u_L = 0.75, p_L = 1.;
  const double rho_R = 0.125, u_R = 0., p_R = 0.1;

  HyperbolicSystem hyperbolic_system;

  ExactRiemannSolution<Description, dim, double> exact_riemann_solution(
      hyperbolic_system, "");

  WaveSpeedEstimator<double> wave_speed_estimator(hyperbolic_system);

  {
    std::stringstream parameters;
    parameters << "subsection exact riemann solution\n"
               << "set primitive state left  = " << rho_L << ", " << u_L << ", "
               << p_L << "\n"
               << "set primitive state right = " << rho_R << ", " << u_R << ", "
               << p_R << "\n"
               << "end\n"
               << "subsection WaveSpeedEstimator\n"
               << "set newton max iterations = 0\n"
               << "end\n"
               << std::endl;
    ParameterAcceptor::initialize(parameters);
  }

  std::cout << std::setprecision(16);
  std::cout << std::scientific;

  /*
   * Exact solution: recover lambda_left_minus and lambda_right_plus from
   * compute() alone (see bisect_wave_speed() above). +-1e6 is comfortably
   * outside the wave speeds of this (or any reasonable) test problem, so
   * compute() there is bit-identical to the untouched far-field state.
   */

  const auto far_left = exact_riemann_solution.compute(Point<dim>{-1.e6}, 1.);
  const auto far_right = exact_riemann_solution.compute(Point<dim>{1.e6}, 1.);

  const double lambda_left_minus =
      bisect_wave_speed<dim>(exact_riemann_solution, far_left, -1.e6, 1.e6);
  const double lambda_right_plus =
      bisect_wave_speed<dim>(exact_riemann_solution, far_right, -1.e6, 1.e6);

  const double lambda_max_exact =
      std::max({-lambda_left_minus, lambda_right_plus, 0.});

  std::cout << "lambda_left_minus (exact)  = " << lambda_left_minus
            << std::endl;
  std::cout << "lambda_right_plus (exact)  = " << lambda_right_plus
            << std::endl;
  std::cout << "lambda_max        (exact)  = " << lambda_max_exact << std::endl;

  /*
   * Approximate solution: WaveSpeedEstimator::compute() takes the
   * expanded primitive Riemann data [rho, u, p, a] directly, see
   * riemann_data_from_state().
   */

  const auto view = hyperbolic_system.view<dim, double>();
  const auto gamma = view.gamma();

  const auto riemann_data = [&](double rho, double u, double p) {
    return std::array<double, 4>{{rho, u, p, std::sqrt(gamma * p / rho)}};
  };

  const auto wave_speed_estimator_view =
      wave_speed_estimator.view<dim, double>();

  const double lambda_max_approximate = wave_speed_estimator_view.compute(
      riemann_data(rho_L, u_L, p_L), riemann_data(rho_R, u_R, p_R));

  std::cout << "lambda_max        (approx) = " << lambda_max_approximate
            << std::endl;
  std::cout << "ratio approx / exact        = "
            << lambda_max_approximate / lambda_max_exact << std::endl;

  /*
   * WaveSpeedEstimator::compute() is a *rigorous upper bound* on the true
   * lambda_max, see @cite GuermondPopov2016b, Section 4. It must not fall
   * below the exact value, and (empirically, for this problem) it should
   * not overshoot it by much either -- a large overshoot would indicate
   * that one of the two disagrees for the wrong reason (e.g., a
   * regression of the u_L != u_R defect this test guards against, which
   * would move lambda_max_exact rather than lambda_max_approximate).
   */

  AssertThrow(lambda_max_approximate >= lambda_max_exact - 1.e-10,
              ExcMessage("WaveSpeedEstimator::compute() must return an "
                         "upper bound on the true maximal wave speed"));
  AssertThrow(lambda_max_approximate <= 1.05 * lambda_max_exact,
              ExcMessage("WaveSpeedEstimator::compute() overestimates "
                         "lambda_max by more than 5%; check whether "
                         "lambda_max_exact regressed instead"));

  /*
   * Reference values (computed independently from the closed-form
   * expressions @cite GuermondPopov2016b, (3.3), (3.7), (3.8), by
   * bisecting phi(p) to machine precision -- not by running ryujin).
   * Before the compute_pstar() fix, p_star (and hence both wave speeds)
   * came out wrong for this u_L != u_R problem.
   */

  AssertThrow(std::abs(lambda_left_minus - (-0.4332159566199232)) < 1.e-9,
              ExcMessage("lambda_left_minus does not match the independent "
                         "reference value for Toro Test 1"));
  AssertThrow(std::abs(lambda_right_plus - 2.1532343675648997) < 1.e-9,
              ExcMessage("lambda_right_plus does not match the independent "
                         "reference value for Toro Test 1"));

  return 0;
}
