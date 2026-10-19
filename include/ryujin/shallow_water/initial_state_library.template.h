//
// SPDX-License-Identifier: Apache-2.0
// [LANL Copyright Statement]
// Copyright (C) 2023 - 2026 by the ryujin authors
// Copyright (C) 2023 - 2024 by Triad National Security, LLC
//

#pragma once

#include <ryujin/shallow_water/hyperbolic_description.h>

#include <ryujin/shallow_water/initial_state_library_shallow_water.h>
#include <ryujin/typed/initial_state_library.h>

namespace ryujin
{
  using HyperbolicDescription = ShallowWater::HyperbolicDescription;

  template <int dim, typename Number>
  class InitialStateLibrary<HyperbolicDescription, dim, Number>
  {
  public:
    using HyperbolicSystem = typename HyperbolicDescription::HyperbolicSystem;

    using View = typename HyperbolicSystem::template View<dim, Number>;

    using initial_state_list_type = std::set<
        std::unique_ptr<InitialState<HyperbolicDescription, dim, Number>>>;

    static void
    populate_initial_state_list(initial_state_list_type &initial_state_list,
                                const HyperbolicSystem &h,
                                const Interface::ParabolicSystem & /*p*/,
                                const std::string &s)
    {
      ShallowWaterInitialStates::
          populate_initial_state_list<HyperbolicDescription, dim, Number>(
              initial_state_list, h, s);
    }
  };
} // namespace ryujin
