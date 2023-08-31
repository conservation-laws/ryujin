//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#include "flux_library.h"

#include "flux_burgers.h"
#include "flux_function.h"
#include "flux_kpp.h"

namespace ryujin
{
  namespace FluxLibrary
  {
    /**
     * Populate a given container with all equation of states defined in
     * this namespace.
     *
     * @ingroup EulerEquations
     */

    void populate_flux_list(
        flux_list_type &flux_list,
        const std::string &subsection)
    {
      auto add = [&](auto &&object) {
        flux_list.emplace(std::move(object));
      };

      add(std::make_shared<Burgers>(subsection));
      add(std::make_shared<Function>(subsection));
      add(std::make_shared<KPP>(subsection));
    }
  } // namespace EquationOfStateLibrary
} // namespace ryujin
