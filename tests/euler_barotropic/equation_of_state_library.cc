#include <barotropic_equation_of_state_isentropic.h>
#include <barotropic_equation_of_state_isothermal.h>
#include <barotropic_equation_of_state_library.h>

#include <deal.II/base/array_view.h>
#include <deal.II/lac/vector.h>

#include <algorithm>
#include <iomanip>
#include <iostream>

/*
 * Test the barotropic EOS library:
 */

using namespace ryujin::BarotropicEquationOfStateLibrary;
using namespace ryujin;
using namespace dealii;

void test(
    const ryujin::BarotropicEquationOfStateLibrary::BarotropicEquationOfState
        &eos)
{
  const auto print_array =
      [](const std::string &name, const auto array, auto &ostream) {
        ostream << name << " =";
        for (const auto &it : array)
          ostream << " " << it;
        ostream << std::endl;
      };

  std::cout << std::setprecision(10);
  std::cout << std::scientific;
  std::cout << "name = " << eos.name() << std::endl;

  {
    const auto rho = 1.4;
    const auto p = eos.pressure(rho);
    const auto e = eos.specific_internal_energy(rho);
    const auto c = eos.speed_of_sound(rho);

    std::cout << "input rho      = " << rho << std::endl //
              << "check p        = " << p << std::endl   //
              << "check e        = " << e << std::endl   //
              << "check c        = " << c << std::endl;
  }

  {
    std::array<double, 5> rho{{1.4, 1.3, 1.2, 1.1, 1.0}};

    std::array<double, 5> p;
    std::ranges::transform(std::begin(rho),
                           std::end(rho),
                           std::begin(p),
                           [&](double rho) { return eos.pressure(rho); });

    std::array<double, 5> e;
    std::ranges::transform(
        std::begin(rho), std::end(rho), std::begin(e), [&](double rho) {
          return eos.specific_internal_energy(rho);
        });

    std::array<double, 5> c;
    std::ranges::transform(std::begin(rho),
                           std::end(rho),
                           std::begin(c),
                           [&](double rho) { return eos.speed_of_sound(rho); });


    print_array("input rho     ", rho, std::cout);
    print_array("check p       ", p, std::cout);
    print_array("check e       ", e, std::cout);
    print_array("check c       ", c, std::cout);
  }
}

int main()
{
  /* isothermal */

  std::cout << "\nIsothermal with c=2" << std::endl;
  Isothermal isothermal("");
  test(isothermal);

  /* isentropic */

  std::cout << "\nIsentropic with k=1, gamma=1.4" << std::endl;
  Isentropic isentropic("");
  test(isentropic);

  return 0;
}
