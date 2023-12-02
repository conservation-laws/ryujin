// force distinct symbols in test
#define Euler EulerTest

#include <hyperbolic_system.h>
#include <multicomponent_vector.h>
#include <simd.h>

#include <limiter.h>

#define CHECK_BOUNDS
#define DEBUG_OUTPUT
#define DEBUG_OUTPUT_LIMITER
#include <limiter.template.h>

using namespace ryujin::Euler;
using namespace ryujin;
using namespace dealii;

int main()
{
  constexpr int dim = 1;

  HyperbolicSystem hyperbolic_system;

  const double relaxation_factor = 1.;
  const double newton_tolerance = 1.e-10;
  const unsigned int newton_max_iter = 2;

  using state_type = HyperbolicSystem::View<dim, double>::state_type;

  using bounds_type = Limiter<dim, double>::Bounds;

  static constexpr unsigned int n_precomputed_values =
      HyperbolicSystem::View<dim, double>::n_precomputed_values;

  using precomputed_type = MultiComponentVector<double, n_precomputed_values>;
  precomputed_type dummy;

  Limiter<dim, double> limiter(hyperbolic_system,
                               dummy,
                               relaxation_factor,
                               newton_tolerance,
                               newton_max_iter);

  const auto view = hyperbolic_system.template view<dim, double>();

  const auto test =
      [&](const state_type &U, const state_type &P, const bounds_type &bounds) {
        std::cout << "State: " << U
                  << "\nSpecific entropy: " << view.specific_entropy(U)
                  << "\nBounds: " << bounds[0] << " " << bounds[1] << " "
                  << bounds[2] << std::endl;

        const auto &[l, success] = limiter.limit(bounds, U, P);

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
    auto U = state_type{{0.8, 1.4, 3.0}};
    auto P = state_type{{-0.1, 0.1, 0.1}};
    auto bounds = bounds_type{0.9, 1.1, 2.0};
    test(U, P, bounds);

    std::cout << "\nMinimum density violation (eps):" << std::endl;
    U = state_type{{0.9 - 1.0e-10, 1.4, 3.0}};
    P = state_type{{-1.0e-20, 0.1, 0.1}};
    bounds = bounds_type{0.9, 1.1, 2.0};
    test(U, P, bounds);

    std::cout << "\nMaximum density violation:" << std::endl;
    U = state_type{{1.2, 1.4, 3.0}};
    P = state_type{{0.1, 0.1, 0.1}};
    bounds = bounds_type{0.9, 1.1, 2.0};
    test(U, P, bounds);

    std::cout << "\nMaximum density violation (eps):" << std::endl;
    U = state_type{{1.1 + 1.0e-10, 1.4, 3.0}};
    P = state_type{{1.0e-20, 0.1, 0.1}};
    bounds = bounds_type{0.9, 1.1, 2.0};
    test(U, P, bounds);

    std::cout << "\nMinimum entropy violation:" << std::endl;
    U = state_type{{1.0, 1.4, 2.8}};
    P = state_type{{0.1, 0.1, -0.1}};
    bounds = bounds_type{0.9, 1.1, 2.0};
    test(U, P, bounds);

    std::cout << "\nMinimum entropy violation (eps):" << std::endl;
    U = state_type{{1.0, 1.4, 2.8}};
    P = state_type{{0.1, 0.1, -1.0e-20}};
    bounds = bounds_type{0.9, 1.1, 1.82 + 1.e-10};
    test(U, P, bounds);
  }

  {
    std::cout << "\n\nChecking individual limiter components:" << std::endl;

    std::cout << "\nMinimum density bound" << std::endl;
    auto U = state_type{{1.0, 1.4, 3.0}};
    auto P = state_type{{-0.2, 0.1, 0.1}};
    auto bounds = bounds_type{0.9, 1.1, 2.0};
    test(U, P, bounds);

    std::cout << "\nMinimum density bound (eps):" << std::endl;
    U = state_type{{0.9 + 1.0e-10, 1.4, 3.0}};
    P = state_type{{-5.0e-10, 0.1, 0.1}};
    bounds = bounds_type{0.9, 1.1, 2.0};
    test(U, P, bounds);

    std::cout << "\nMaximum density bound" << std::endl;
    U = state_type{{1.0, 1.4, 3.0}};
    P = state_type{{0.2, 0.1, 0.1}};
    bounds = bounds_type{0.9, 1.1, 1.0};
    test(U, P, bounds);

    std::cout << "\nMaximum density bound (eps):" << std::endl;
    U = state_type{{1.1 - 1.0e-10, 1.4, 3.0}};
    P = state_type{{5.0e-10, 0.1, 0.1}};
    bounds = bounds_type{0.9, 1.1, 1.0};
    test(U, P, bounds);

    std::cout << "\nMinimum entropy bound" << std::endl;
    U = state_type{{1.0, 1.4, 2.8}};
    P = state_type{{0.1, 0.1, -0.3}};
    bounds = bounds_type{0.9, 1.1, 1.8};
    test(U, P, bounds);

    std::cout << "\nMinimum entropy bound (eps):" << std::endl;
    U = state_type{{1.0, 1.4, 2.8}};
    P = state_type{{0.1, 0.1, -4.0e-10}};
    bounds = bounds_type{0.9, 1.1, 1.82 - 1.e-10};
    test(U, P, bounds);
  }

  return 0;
}
