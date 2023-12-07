//
// SPDX-License-Identifier: MIT or BSD-3-Clause
// [LANL Copyright Statement]
// Copyright (C) 2020 - 2023 by the ryujin authors
// Copyright (C) 2023 - 2023 by Triad National Security, LLC
//

#pragma once

#include "description.h"

#include "initial_state_library_shallow_water.h"
#include <initial_state_library.h>

namespace ryujin
{
  using Description = ShallowWater::Description;

  template <int dim, typename Number>
  class InitialStateLibrary<Description, dim, Number>
  {
  public:
    using HyperbolicSystem = typename Description::HyperbolicSystem;

    using HyperbolicSystemView =
        typename HyperbolicSystem::template View<dim, Number>;

    using initial_state_list_type =
        std::set<std::unique_ptr<InitialState<Description, dim, Number>>>;

    static void
    populate_initial_state_list(initial_state_list_type &initial_state_list,
                                const HyperbolicSystem &h,
                                const std::string &s)
    {
      ShallowWaterInitialStates::
          populate_initial_state_list<Description, dim, Number>(
              initial_state_list, h, s);
    }
  };
} // namespace ryujin
