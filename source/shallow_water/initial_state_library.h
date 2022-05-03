//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include <hyperbolic_system.h>

#include <initial_state.h>

namespace ryujin
{
  namespace InitialStateLibrary
  {
    /**
     * Populate a given container with all initial state defined in this
     * namespace
     *
     * @ingroup InitialValues
     */
    template <int dim, typename Number, typename T>
    void populate_initial_state_list(T &initial_state_list,
                                     const HyperbolicSystem &hyperbolic_system,
                                     const std::string &subsection)
    {
//       using N = Number;
//       const auto &h = hyperbolic_system;
//       const auto &s = subsection;
//       initial_state_list.emplace(std::make_unique<Uniform<dim, N>>(h, s));
    }

  } // namespace InitialStateLibrary
} // namespace ryujin
