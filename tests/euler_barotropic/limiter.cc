//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

// force distinct symbols in test
#define EulerBarotropic EulerBarotropicTest

#include <ryujin/base/simd.h>
#include <ryujin/euler_barotropic/hyperbolic_system.h>
#include <ryujin/linear_algebra/multicomponent_vector.h>

#define DEBUG_EXPENSIVE_BOUNDS_CHECK
#define DEBUG_OUTPUT
#define DEBUG_OUTPUT_LIMITER
#include <ryujin/euler_barotropic/limiter.h>

using namespace ryujin::EulerBarotropic;
using namespace ryujin;
using namespace dealii;

int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  constexpr int dim = 1;

  HyperbolicSystem hyperbolic_system;
  Limiter<double> limiter(hyperbolic_system);

  using state_type = HyperbolicSystemView<dim, double>::state_type;

  using bounds_type = LimiterView<dim, double>::Bounds;

  const auto limiter_view = limiter.view<dim, double>();

  const auto view = hyperbolic_system.template view<dim, double>();

  const auto test =
      [&](const state_type &U, const state_type &P, const bounds_type &bounds) {
        const auto rho = view.density(U);
        const auto v2 = view.momentum(U).norm_square() / (rho * rho);
        std::cout << "State: " << U << "\nDensity: " << rho
                  << "\nSquare of the velocity: " << v2
                  << "\nBounds: " << bounds[0] << " " << bounds[1] << " "
                  << bounds[2] << std::endl;

        const auto &[l, success] = limiter_view.limit(bounds, U, P);

        std::cout << "l: " << l;
        if (success)
          std::cout << "\nSuccess!";
        else
          std::cout << "\nFailure!";
        std::cout << std::endl;
      };

  std::cout << std::setprecision(16);
  std::cout << std::scientific;

  {
    std::cout << "\n\nChecking exceptional cases:" << std::endl;

    std::cout << "\nMinimum density violation:" << std::endl;
    auto U = state_type{{0.8, 1.4}};
    auto P = state_type{{-0.1, 0.1}};
    auto bounds = bounds_type{0.9, 1.1, 9.0};
    test(U, P, bounds);

    std::cout << "\nMinimum density violation (eps):" << std::endl;
    U = state_type{{0.9 - 1.0e-10, 1.4}};
    P = state_type{{-1.0e-20, 0.1}};
    bounds = bounds_type{0.9, 1.1, 9.0};
    test(U, P, bounds);

    std::cout << "\nMaximum density violation:" << std::endl;
    U = state_type{{1.2, 1.4}};
    P = state_type{{0.1, 0.1}};
    bounds = bounds_type{0.9, 1.1, 9.0};
    test(U, P, bounds);

    std::cout << "\nMaximum density violation (eps):" << std::endl;
    U = state_type{{1.1 + 1.0e-10, 1.4}};
    P = state_type{{1.0e-20, 0.1}};
    bounds = bounds_type{0.9, 1.1, 9.0};
    test(U, P, bounds);

    std::cout << "\nMaximum velocity violation:" << std::endl;
    U = state_type{{1.0, 3.0}};
    P = state_type{{0.0, 0.1}};
    bounds = bounds_type{0.5, 1.5, 4.0};
    test(U, P, bounds);

    std::cout << "\nMaximum velocity violation (eps):" << std::endl;
    U = state_type{{1.0, 2.0 + 1.0e-10}};
    P = state_type{{0.0, 1.0e-20}};
    bounds = bounds_type{0.5, 1.5, 4.0};
    test(U, P, bounds);
  }

  {
    std::cout << "\n\nChecking individual limiter components:" << std::endl;

    std::cout << "\nMinimum density bound" << std::endl;
    auto U = state_type{{1.0, 1.4}};
    auto P = state_type{{-0.2, 0.1}};
    auto bounds = bounds_type{0.9, 1.1, 9.0};
    test(U, P, bounds);

    std::cout << "\nMinimum density bound (eps):" << std::endl;
    U = state_type{{0.9 + 1.0e-10, 1.4}};
    P = state_type{{-5.0e-10, 0.1}};
    bounds = bounds_type{0.9, 1.1, 9.0};
    test(U, P, bounds);

    std::cout << "\nMaximum density bound" << std::endl;
    U = state_type{{1.0, 1.4}};
    P = state_type{{0.2, 0.1}};
    bounds = bounds_type{0.9, 1.1, 9.0};
    test(U, P, bounds);

    std::cout << "\nMaximum density bound (eps):" << std::endl;
    U = state_type{{1.1 - 1.0e-10, 1.4}};
    P = state_type{{5.0e-10, 0.1}};
    bounds = bounds_type{0.9, 1.1, 9.0};
    test(U, P, bounds);

    std::cout << "\nMaximum velocity bound" << std::endl;
    U = state_type{{1.0, 1.0}};
    P = state_type{{0.0, 3.0}};
    bounds = bounds_type{0.5, 1.5, 4.0};
    test(U, P, bounds);

    std::cout << "\nMaximum velocity bound (eps):" << std::endl;
    U = state_type{{1.0, 2.0 - 1.0e-10}};
    P = state_type{{0.0, 5.0e-10}};
    bounds = bounds_type{0.5, 1.5, 4.0};
    test(U, P, bounds);

    std::cout << "\nMaximum velocity bound near vacuum" << std::endl;
    U = state_type{{1.0e-6, 1.0e-6}};
    P = state_type{{0.0, 1.0e-4}};
    bounds = bounds_type{1.0e-7, 1.0e-5, 1.21};
    test(U, P, bounds);
  }

  return 0;
}
