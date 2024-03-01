//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include "equation_of_state.h"

namespace ryujin
{
  namespace EquationOfStateLibrary
  {
    using equation_of_state_list_type =
        std::set<std::shared_ptr<EquationOfState>>;

    /**
     * Populate a given container with all equation of states defined in
     * this namespace.
     *
     * @ingroup EulerEquations
     */
    void populate_equation_of_state_list(
        equation_of_state_list_type &equation_of_state_list,
        const std::string &subsection);

  } // namespace EquationOfStateLibrary
} // namespace ryujin
