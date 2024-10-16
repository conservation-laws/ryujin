// force distinct symbols in test
#define EulerAEOS EulerAEOSTest

#include <hyperbolic_system.h>
#include <multicomponent_vector.h>
#include <simd.h>

#include <limiter.h>

#define EXPENSIVE_BOUNDS_CHECK
#define DEBUG_OUTPUT
#define DEBUG_OUTPUT_LIMITER
#include <limiter.template.h>

using namespace ryujin::EulerAEOS;
using namespace ryujin;
using namespace dealii;

int main()
{
  constexpr int dim = 1;

  HyperbolicSystem hyperbolic_system;
  Limiter<dim, double>::Parameters limiter_parameters;

  const auto set_covolume = [&](const double covolume) {
    /*
     * Set the interpolatory covolume by selecting an equation of state
     * with a covolume.
     */
    std::stringstream parameters;
    parameters << "subsection HyperbolicSystem\n"
               << "set equation of state = noble abel stiffened gas\n"
               << "subsection noble abel stiffened gas\n"
               << "set covolume b = " << std::to_string(covolume) << "\n"
               << "set reference pressure = 0.1\n"
               << "set reference specific internal energy = 0.1\n"
               << "end\n"
               << "end\n"
               << std::endl;
    ParameterAcceptor::initialize(parameters);
  };

  using state_type = HyperbolicSystemView<dim, double>::state_type;

  using bounds_type = Limiter<dim, double>::Bounds;

  static constexpr unsigned int n_precomputed_values =
      HyperbolicSystemView<dim, double>::n_precomputed_values;

  using precomputed_type =
      Vectors::MultiComponentVector<double, n_precomputed_values>;
  precomputed_type dummy;

  Limiter<dim, double> limiter(hyperbolic_system, limiter_parameters, dummy);

  const auto view = hyperbolic_system.template view<dim, double>();

  constexpr double gamma = 1.4;

  const auto test =
      [&](const state_type &U, const state_type &P, const bounds_type &bounds) {
        std::cout << "State: " << U << "\nSpecific entropy: "
                  << view.surrogate_specific_entropy(U, gamma)
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
    std::cout << "\n\nChecking individual limiter components:" << std::endl;

    std::cout << "\nMinimum density bound" << std::endl;
    auto U = state_type{{1.0, 1.4, 3.0}};
    auto P = state_type{{-0.2, 0.1, 0.1}};
    auto bounds = bounds_type{0.9, 1.1, 2.0, gamma};
    test(U, P, bounds);

    std::cout << "\nMinimum density bound (eps):" << std::endl;
    U = state_type{{0.9 + 1.0e-10, 1.4, 3.0}};
    P = state_type{{-5.0e-10, 0.1, 0.1}};
    bounds = bounds_type{0.9, 1.1, 2.0, gamma};
    test(U, P, bounds);

    std::cout << "\nMaximum density bound" << std::endl;
    U = state_type{{1.0, 1.4, 3.0}};
    P = state_type{{0.2, 0.1, 0.1}};
    bounds = bounds_type{0.9, 1.1, 1.0, gamma};
    test(U, P, bounds);

    std::cout << "\nMaximum density bound (eps):" << std::endl;
    U = state_type{{1.1 - 1.0e-10, 1.4, 3.0}};
    P = state_type{{5.0e-10, 0.1, 0.1}};
    bounds = bounds_type{0.9, 1.1, 1.0, gamma};
    test(U, P, bounds);

    std::cout << "\nMinimum entropy bound" << std::endl;
    U = state_type{{1.0, 1.4, 2.8}};
    P = state_type{{0.1, 0.1, -0.3}};
    bounds = bounds_type{0.9, 1.1, 1.8, gamma};
    test(U, P, bounds);

    std::cout << "\nMinimum entropy bound (eps):" << std::endl;
    U = state_type{{1.0, 1.4, 2.8}};
    P = state_type{{0.1, 0.1, -4.0e-10}};
    bounds = bounds_type{0.9, 1.1, 1.82 - 1.e-10, gamma};
    test(U, P, bounds);
  }

  {
    std::cout << "\n\nChecking individual limiter components with covolume:"
              << std::endl;

    set_covolume(1.0e-1);
    std::cout << "Covolume b = 1.0e-1" << std::endl;

    std::cout << "\nMinimum density bound" << std::endl;
    auto U = state_type{{1.0, 1.4, 3.0}};
    auto P = state_type{{-0.2, 0.1, 0.1}};
    auto bounds = bounds_type{0.9, 1.1, 1.7, gamma};
    test(U, P, bounds);

    std::cout << "\nMinimum density bound (eps):" << std::endl;
    U = state_type{{0.9 + 1.0e-10, 1.4, 3.0}};
    P = state_type{{-5.0e-10, 0.1, 0.1}};
    bounds = bounds_type{0.9, 1.1, 2.0, gamma};
    test(U, P, bounds);

    std::cout << "\nMaximum density bound" << std::endl;
    U = state_type{{1.0, 1.4, 3.0}};
    P = state_type{{0.2, 0.1, 0.1}};
    bounds = bounds_type{0.9, 1.1, 1.0, gamma};
    test(U, P, bounds);

    std::cout << "\nMaximum density bound (eps):" << std::endl;
    U = state_type{{1.1 - 1.0e-10, 1.4, 3.0}};
    P = state_type{{5.0e-10, 0.1, 0.1}};
    bounds = bounds_type{0.9, 1.1, 1.0, gamma};
    test(U, P, bounds);

    std::cout << "\nMinimum entropy bound" << std::endl;
    U = state_type{{1.0, 1.4, 2.8}};
    P = state_type{{0.1, 0.1, -0.3}};
    bounds = bounds_type{0.9, 1.1, 1.5, gamma};
    test(U, P, bounds);

    std::cout << "\nMinimum entropy bound (eps):" << std::endl;
    U = state_type{{1.0, 1.4, 2.8}};
    P = state_type{{0.1, 0.1, -4.0e-10}};
    bounds = bounds_type{0.9, 1.1, 1.7448913582358123 - 1.e-10, gamma};
    test(U, P, bounds);
  }

  {
    std::cout << "\n\nChecking for limiting close to maximal compressibility:"
              << std::endl;

    const double b = 0.2;
    set_covolume(0.2);

    std::cout << "\nMaximum density bound:" << std::endl;
    double rho_max = 1 / b - 1.0e-6;
    double rho_limit = (gamma + 1) * rho_max / (gamma - 1 + 2 * b * rho_max);
    auto U = state_type{{4.5, 1.4, 100000.0}};
    auto P = state_type{{1.0, 0.1, 0.1}};
    auto bounds = bounds_type{0.9, rho_limit, 1.6, gamma};
    test(U, P, bounds);

    rho_max = 1 / b - 1.0e-3;
    rho_limit = (gamma + 1) * rho_max / (gamma - 1 + 2 * b * rho_max);
    std::cout << "\nMaximum density and entropy bounds:" << std::endl;
    U = state_type{{4.5, 1.4, 500.0}};
    P = state_type{{1.0, 0.1, 0.1}};
    bounds = bounds_type{0.9, rho_limit, 1.6, gamma};
    test(U, P, bounds);
  }

  return 0;
}
