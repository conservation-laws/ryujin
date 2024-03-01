//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include "flux.h"

namespace ryujin
{
  namespace FluxLibrary
  {
    using flux_list_type = std::set<std::shared_ptr<Flux>>;

    /**
     * Populate a given container with all fluxes defined in this
     * namespace.
     *
     * @ingroup ScalarConservation
     */
    void populate_flux_list(flux_list_type &flux_list,
                            const std::string &subsection);

  } // namespace FluxLibrary
} // namespace ryujin
