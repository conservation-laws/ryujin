//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#include <ryujin/euler_aeos/equation_of_state_bumpy_barotropic_pressure.h>
#include <ryujin/euler_aeos/equation_of_state_function.h>
#include <ryujin/euler_aeos/equation_of_state_hayes.h>
#include <ryujin/euler_aeos/equation_of_state_jones_wilkins_lee.h>
#include <ryujin/euler_aeos/equation_of_state_library.h>
#include <ryujin/euler_aeos/equation_of_state_noble_abel_stiffened_gas.h>
#include <ryujin/euler_aeos/equation_of_state_polytropic_gas.h>
#include <ryujin/euler_aeos/equation_of_state_pressureless.h>
#include <ryujin/euler_aeos/equation_of_state_sesame.h>
#include <ryujin/euler_aeos/equation_of_state_simple_macaw.h>
#include <ryujin/euler_aeos/equation_of_state_van_der_waals.h>

namespace ryujin
{
  template <>
  auto EquationOfStateLibrary::defaults() -> std::vector<Creator>
  {
    using namespace EquationsOfState;
    return {
        creator<BumpyBarotropicPressure>(),
        creator<Function>(),
        creator<Hayes>(),
        creator<JonesWilkinsLee>(),
        creator<NobleAbelStiffenedGas>(),
        creator<PolytropicGas>(),
        creator<Sesame>(),
        creator<SimpleMacaw>(),
        creator<VanDerWaals>(),
        creator<Pressureless>(),
    };
  }

  /* instantiations */

  template class Library<EquationOfState, const std::string &>;
} // namespace ryujin
