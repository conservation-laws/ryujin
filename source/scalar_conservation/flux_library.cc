//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 by the ryujin authors
//

#include <ryujin/scalar_conservation/flux_library.h>

#include <ryujin/scalar_conservation/flux_burgers.h>
#include <ryujin/scalar_conservation/flux_function.h>
#include <ryujin/scalar_conservation/flux_kpp.h>

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

    void populate_flux_list(flux_list_type &flux_list,
                            const std::string &subsection)
    {
      auto add = [&](auto &&object) { flux_list.emplace(std::move(object)); };

      add(std::make_shared<Burgers>(subsection));
      add(std::make_shared<Function>(subsection));
      add(std::make_shared<KPP>(subsection));
    }
  } // namespace FluxLibrary
} // namespace ryujin
