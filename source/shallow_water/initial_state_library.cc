//
// SPDX-License-Identifier: Apache-2.0
// [LANL Copyright Statement]
// Copyright (C) 2023 - 2026 by the ryujin authors
// Copyright (C) 2023 - 2024 by Triad National Security, LLC
//

#include <ryujin/shallow_water/initial_state_library.h>
#include <shallow_water/initial_state_defaults.h>

namespace ryujin
{
  template <>
  auto InitialStateLibrary<ShallowWater::HyperbolicDescription, 1, NUMBER>::
      defaults() -> std::vector<Creator>
  {
    return ShallowWaterInitialStates::
        defaults<ShallowWater::HyperbolicDescription, 1, NUMBER>();
  }

  template <>
  auto InitialStateLibrary<ShallowWater::HyperbolicDescription, 2, NUMBER>::
      defaults() -> std::vector<Creator>
  {
    return ShallowWaterInitialStates::
        defaults<ShallowWater::HyperbolicDescription, 2, NUMBER>();
  }

  template <>
  auto InitialStateLibrary<ShallowWater::HyperbolicDescription, 3, NUMBER>::
      defaults() -> std::vector<Creator>
  {
    return ShallowWaterInitialStates::
        defaults<ShallowWater::HyperbolicDescription, 3, NUMBER>();
  }

  /* instantiations */

  template class Library<
      InitialState<ShallowWater::HyperbolicDescription, 1, NUMBER>,
      const ShallowWater::HyperbolicSystem &,
      const std::string &>;
  template class Library<
      InitialState<ShallowWater::HyperbolicDescription, 2, NUMBER>,
      const ShallowWater::HyperbolicSystem &,
      const std::string &>;
  template class Library<
      InitialState<ShallowWater::HyperbolicDescription, 3, NUMBER>,
      const ShallowWater::HyperbolicSystem &,
      const std::string &>;
} // namespace ryujin
