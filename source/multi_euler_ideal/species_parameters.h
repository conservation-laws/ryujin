//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2025 by the ryujin authors
// Copyright (C) 2025 by Triad National Security, LLC
//

#pragma once

namespace ryujin
{
  namespace MultiSpeciesEuler
  {
    /**
     * Compile-time parameter specifying the number of species in the
     * multi-species Euler equations. Change this value and recompile to
     * use a different number of species.
     *
     * The state vector dimension will be: n_species + dim + 1
     * (n_species partial densities + dim momentum components + 1 total energy)
     *
     * @note n_species must be at most 3. For n_species >= 4 the limiter
     * bounds array becomes too large. FIXME.
     */
    constexpr unsigned int n_species = 2;
    static_assert(n_species >= 1 && n_species <= 3,
                  "n_species must be between 1 and 3");

  } // namespace MultiSpeciesEuler
} // namespace ryujin
