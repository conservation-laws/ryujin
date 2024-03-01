//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#include "equation_of_state_library.h"

#include "equation_of_state_function.h"
#include "equation_of_state_jones_wilkins_lee.h"
#include "equation_of_state_noble_abel_stiffened_gas.h"
#include "equation_of_state_polytropic_gas.h"
#include "equation_of_state_sesame.h"
#include "equation_of_state_van_der_waals.h"

namespace ryujin
{
  namespace EquationOfStateLibrary
  {
    /**
     * Populate a given container with all equation of states defined in
     * this namespace.
     *
     * @ingroup EulerEquations
     */

    void populate_equation_of_state_list(
        equation_of_state_list_type &equation_of_state_list,
        const std::string &subsection)
    {
      auto add = [&](auto &&object) {
        equation_of_state_list.emplace(std::move(object));
      };

      add(std::make_shared<Function>(subsection));
      add(std::make_shared<JonesWilkinsLee>(subsection));
      add(std::make_shared<NobleAbelStiffenedGas>(subsection));
      add(std::make_shared<PolytropicGas>(subsection));
      add(std::make_shared<Sesame>(subsection));
      add(std::make_shared<VanDerWaals>(subsection));
    }
  } // namespace EquationOfStateLibrary
} // namespace ryujin
