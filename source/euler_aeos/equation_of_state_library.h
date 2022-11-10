//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "equation_of_state.h"

#include "equation_of_state_jwl.h"
#include "equation_of_state_nasg_gas.h"
#include "equation_of_state_polytropic_gas.h"

namespace ryujin
{
  namespace EquationOfStateLibrary
  {
    /**
     * Populate a given container with all equation of states defined in
     * this namespace.
     *
     * @ingroup EquationOfState
     */
    template <typename T>
    void populate_equation_of_state_list(T &equation_of_state_list,
                                         const std::string &subsection)
    {
      auto add = [&](auto &&object) {
        equation_of_state_list.emplace(std::move(object));
      };

      add(std::make_unique<PolytropicGas>(subsection));
      add(std::make_unique<NobleAbleStiffenedGas>(subsection));
      add(std::make_unique<JonesWilkinsLee>(subsection));
    }

  } // namespace EquationOfStateLibrary
} // namespace ryujin
