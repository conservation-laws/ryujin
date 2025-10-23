//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 by the ryujin authors
//

#include "barotropic_equation_of_state_library.h"

#include "barotropic_equation_of_state_function.h"
#include "barotropic_equation_of_state_isentropic.h"
#include "barotropic_equation_of_state_isothermal.h"
#include "barotropic_equation_of_state_pressureless.h"

namespace ryujin
{
  namespace BarotropicEquationOfStateLibrary
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
      add(std::make_shared<Isentropic>(subsection));
      add(std::make_shared<Isothermal>(subsection));
      add(std::make_shared<Pressureless>(subsection));
    }
  } // namespace BarotropicEquationOfStateLibrary
} // namespace ryujin
