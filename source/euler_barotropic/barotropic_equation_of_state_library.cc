//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#include <ryujin/euler_barotropic/barotropic_equation_of_state_function.h>
#include <ryujin/euler_barotropic/barotropic_equation_of_state_isentropic.h>
#include <ryujin/euler_barotropic/barotropic_equation_of_state_isothermal.h>
#include <ryujin/euler_barotropic/barotropic_equation_of_state_library.h>
#include <ryujin/euler_barotropic/barotropic_equation_of_state_pressureless.h>

namespace ryujin
{
  template <>
  auto BarotropicEquationOfStateLibrary::defaults() -> std::vector<Creator>
  {
    using namespace BarotropicEquationsOfState;
    return {
        creator<Function>(),
        creator<Isentropic>(),
        creator<Isothermal>(),
        creator<Pressureless>(),
    };
  }

  /* instantiations */

  template class Library<BarotropicEquationOfState, const std::string &>;
} // namespace ryujin
