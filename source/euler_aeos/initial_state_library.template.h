//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include "hyperbolic_description.h"

#include "../euler/initial_state_library_euler.h"
#include <initial_state_library.h>

namespace ryujin
{
  using HyperbolicDescription = EulerAEOS::HyperbolicDescription;

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
      EulerInitialStates::
          populate_initial_state_list<HyperbolicDescription, dim, Number>(
              initial_state_list, h, s);
    }
  };
} // namespace ryujin
