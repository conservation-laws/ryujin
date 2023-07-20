//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "equation_of_state.h"

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

      using equation_of_state_list_type =
          std::set<std::unique_ptr<EquationOfState>>;

      void populate_equation_of_state_list(
          equation_of_state_list_type &equation_of_state_list,
          const std::string &subsection);

    } // namespace EquationOfStateLibrary
  }   // namespace EulerAEOS
} // namespace ryujin
