//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// [LANL Copyright Statement]
// Copyright (C) 2023 - 2026 by the ryujin authors
// Copyright (C) 2023 - 2024 by Triad National Security, LLC
//

#include <ryujin/shallow_water/initial_state_library.h>
#include <shallow_water/initial_state_defaults.h>

namespace ryujin
{
  template <>
  auto
  InitialStateLibrary<ShallowWater::HyperbolicSystem, 1, NUMBER>::defaults()
      -> std::vector<Creator>
  {
    return ShallowWaterInitialStates::
        defaults<ShallowWater::HyperbolicSystem, 1, NUMBER>();
  }

  template <>
  auto
  InitialStateLibrary<ShallowWater::HyperbolicSystem, 2, NUMBER>::defaults()
      -> std::vector<Creator>
  {
    return ShallowWaterInitialStates::
        defaults<ShallowWater::HyperbolicSystem, 2, NUMBER>();
  }

  template <>
  auto
  InitialStateLibrary<ShallowWater::HyperbolicSystem, 3, NUMBER>::defaults()
      -> std::vector<Creator>
  {
    return ShallowWaterInitialStates::
        defaults<ShallowWater::HyperbolicSystem, 3, NUMBER>();
  }

  /* instantiations */

  template class Library<
      InitialState<ShallowWater::HyperbolicSystem, 1, NUMBER>,
      const ShallowWater::HyperbolicSystem &,
      const std::string &>;
  template class Library<
      InitialState<ShallowWater::HyperbolicSystem, 2, NUMBER>,
      const ShallowWater::HyperbolicSystem &,
      const std::string &>;
  template class Library<
      InitialState<ShallowWater::HyperbolicSystem, 3, NUMBER>,
      const ShallowWater::HyperbolicSystem &,
      const std::string &>;
} // namespace ryujin
