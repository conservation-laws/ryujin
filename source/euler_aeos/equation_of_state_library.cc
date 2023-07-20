//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "equation_of_state_library.h"

#include "equation_of_state_jones_wilkins_lee.h"
#include "equation_of_state_noble_abel_stiffened_gas.h"
#include "equation_of_state_polytropic_gas.h"
#include "equation_of_state_van_der_waals.h"

namespace ryujin
{
  namespace EulerAEOS
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

        add(std::make_unique<PolytropicGas>(subsection));
        add(std::make_unique<VanDerWaals>(subsection));
        add(std::make_unique<NobleAbelStiffenedGas>(subsection));
        add(std::make_unique<JonesWilkinsLee>(subsection));
      }

    } // namespace EquationOfStateLibrary
  }   // namespace EulerAEOS
} // namespace ryujin
