#include <equation_of_state_function.h>
#include <equation_of_state_hayes.h>
#include <equation_of_state_jones_wilkins_lee.h>
#include <equation_of_state_noble_abel_stiffened_gas.h>
#include <equation_of_state_polytropic_gas.h>
#include <equation_of_state_pressureless.h>
#include <equation_of_state_simple_macaw.h>
#include <equation_of_state_van_der_waals.h>

#include <deal.II/base/array_view.h>
#include <deal.II/lac/vector.h>

#include <iomanip>
#include <iostream>

/*
 * Test the EOS library:
 */

using namespace ryujin::EquationOfStateLibrary;
using namespace ryujin;
using namespace dealii;

/*
 * Create struct with default test values. Test values can be modified to
 * accomodate specific EOS.
 */
struct testValues {
  double rho_scalar = 1.4;
  double e_scalar = 1.0 / 1.4 / 0.4;

  std::array<double, 5> rho_array{{1.4, 1.3, 1.2, 1.1, 1.0}};
  std::array<double, 5> e_array{{0.3, 0.2, 0.1, 0.05, 0.025}};
};

void test(const ryujin::EquationOfStateLibrary::EquationOfState &eos,
          const testValues &tv = testValues())
{
  const auto print_array =
      [](const std::string name, const auto array, auto &ostream) {
        ostream << name << " =";
        for (const auto &it : array)
          ostream << " " << it;
        ostream << std::endl;
      };

  std::cout << std::setprecision(10);
  std::cout << std::scientific;
  std::cout << "name = " << eos.name() << std::endl;

  {
    auto rho = tv.rho_scalar;
    auto e = tv.e_scalar;

    const auto p = eos.pressure(rho, e);
    const auto e_back = eos.specific_internal_energy(rho, p);
    const auto s = eos.specific_entropy(rho, e);
    const auto e_min = eos.cold_curve_bound(rho);
    const auto T = eos.temperature(rho, e);
    const auto c = eos.speed_of_sound(rho, e);

    std::cout << "input rho      = " << rho << std::endl    //
              << "input e        = " << e << std::endl      //
              << "output p       = " << p << std::endl      //
              << "check e_back   = " << e_back << std::endl //
              << "check s        = " << s << std::endl      //
              << "check e_min    = " << e_min << std::endl  //
              << "check T        = " << T << std::endl      //
              << "check c        = " << c << std::endl;     //
  }

  {
    auto rho = tv.rho_array;
    auto e = tv.e_array;

    std::array<double, 5> p;
    std::array<double, 5> e_back;
    std::array<double, 5> c;
    std::array<double, 5> T;

    std::transform(
        std::begin(rho),
        std::end(rho),
        std::begin(e),
        std::begin(p),
        [&](const auto rho, const auto e) { return eos.pressure(rho, e); });

    std::transform(std::begin(rho),
                   std::end(rho),
                   std::begin(p),
                   std::begin(e_back),
                   [&](const auto rho, const auto p) {
                     return eos.specific_internal_energy(rho, p);
                   });

    std::transform(std::begin(rho),
                   std::end(rho),
                   std::begin(e),
                   std::begin(c),
                   [&](const auto rho, const auto e) {
                     return eos.speed_of_sound(rho, e);
                   });

    std::transform(
        std::begin(rho),
        std::end(rho),
        std::begin(e),
        std::begin(T),
        [&](const auto rho, const auto e) { return eos.temperature(rho, e); });

    print_array("input rho     ", rho, std::cout);
    print_array("input e       ", e, std::cout);
    print_array("output p      ", p, std::cout);
    print_array("check e_back  ", e_back, std::cout);
    print_array("check c       ", c, std::cout);
    print_array("check T       ", T, std::cout);
  }
}

int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  /* polytropic gas */

  std::cout << "\nPolytropicGas with gamma=1.4" << std::endl;
  PolytropicGas polytropic_gas("");
  test(polytropic_gas);

  /* function */

  std::cout << "\nFunction (polytropic gas with gamma=1.4)" << std::endl;
  EquationOfStateLibrary::Function function("");
  test(function);

  /* noble Abel stiffened gas */

  std::cout << "\nNobleAbelStiffenedGas with gamma=1.4, b=0, q=0, pinf=0"
            << std::endl;
  NobleAbelStiffenedGas noble_abel_stiffened_gas("");
  test(noble_abel_stiffened_gas);

  std::cout
      << "\nNobleAbelStiffenedGas with gamma=1.4, b=0.2, q=0.00125, pinf=0.005"
      << std::endl;
  {
    std::stringstream parameters;
    parameters << "subsection noble abel stiffened gas\n"
               << "set gamma = 1.4\n"
               << "set covolume b = 0.2\n"
               << "set reference specific internal energy = 0.00125\n"
               << "set reference pressure = 0.005\n"
               << "end\n"
               << std::endl;
    ParameterAcceptor::initialize(parameters);
  }
  test(noble_abel_stiffened_gas);

  /* van der Waals */

  std::cout << "\nVanDerWaals with gamma=1.4, a=0, b=0" << std::endl;
  VanDerWaals van_der_waals("");
  test(van_der_waals);

  std::cout << "\nVanDerWaals with gamma=1.4, a=0.015, b=0.2" << std::endl;
  {
    std::stringstream parameters;
    parameters << "subsection van der waals\n"
               << "set gamma = 1.40\n"
               << "set covolume b = 0.2\n"
               << "set vdw a = 0.015\n"
               << "end\n"
               << std::endl;
    ParameterAcceptor::initialize(parameters);
  }
  test(van_der_waals);

  /* Jones Wilkins Lee */

  std::cout << "\nJonesWilkinsLee with omega=0.8938, A=6.3207e13, B=-4.472e9, "
               "R1=11.3, R2=1.13, rho_0=1895, q_0=0"
            << std::endl;
  JonesWilkinsLee jones_wilkins_lee("");
  test(jones_wilkins_lee);

  std::cout << "\nJonesWilkinsLee with omega=0.4, A=0, B=0, "
               "R1=1, R2=1, rho_0=1, q_0=0, c_v=1"
            << std::endl;
  {
    std::stringstream parameters;
    parameters << "subsection jones wilkins lee\n"
               << "set A     = 0\n"
               << "set B     = 0\n"
               << "set R1    = 1\n"
               << "set R2    = 1\n"
               << "set omega = 0.4\n"
               << "set rho_0 = 1\n"
               << "set q_0   = 0\n"
               << "set c_v   = 1\n"
               << "end\n"
               << std::endl;
    ParameterAcceptor::initialize(parameters);
  }
  test(jones_wilkins_lee);

  /* simple macaw */

  testValues macaw_values;
  macaw_values.rho_scalar = 1.86;
  macaw_values.e_scalar = 1.49017421e2;
  macaw_values.rho_array = {2., 2.5, 3., 3.5, 4.0};
  macaw_values.e_array = {1.5e2, 1.4e2, 1.3e2, 1.2e2, 1.1e2};

  std::cout << "\nSimple Macaw with default parameters" << std::endl;
  SimpleMacaw simple_macaw("");
  {
    std::stringstream parameters;
    parameters << "subsection simple macaw\n"
               << "set A                           = 7.3\n"
               << "set B                           = 3.9\n"
               << "set Gamma                       = 0.5\n"
               << "set cvInf                       = 0.000389\n"
               << "set reference T0                = 150\n"
               << "set reference rho0              = 8.952\n"
               << "end\n"
               << std::endl;

    ParameterAcceptor::initialize(parameters);
  }
  test(simple_macaw, macaw_values);

  /* pressureless */

  std::cout << "\nPressureless" << std::endl;
  Pressureless pressureless("");
  test(pressureless);

  /* Hayes */

  testValues hayes_values;
  hayes_values.rho_scalar = 2.0;
  hayes_values.e_scalar = 0.01;
  hayes_values.rho_array = {2.1, 2.2, 2.3, 2.4, 2.5};
  hayes_values.e_array = {0.25, 0.3, 0.35, 0.4, 0.45};

  std::cout << "\nHayes with default parameters" << std::endl;
  Hayes hayes("");
  {
    std::stringstream parameters;
    parameters << "subsection hayes\n"
               << "set N                           = 5.6\n"
               << "set k_0                         = 12.6\n"
               << "set gamma_0                     = 1.0715848\n"
               << "set c_v                         = 1.11e-3\n"
               << "set T_0                         = 298.15\n"
               << "set rho_0                       = 1.844\n"
               << "end\n"
               << std::endl;

    ParameterAcceptor::initialize(parameters);
  }
  test(hayes, hayes_values);

  return 0;
}
