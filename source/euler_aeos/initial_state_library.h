//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"

#include <initial_state.h>

namespace ryujin
{
  namespace EulerAEOS
  {
    struct Description;

    struct InitialStateLibrary {
      template <int dim, typename Number>
      using initial_state_list_type =
          std::set<std::unique_ptr<InitialState<Description, dim, Number>>>;

      /**
       * Populate a given container with all initial state defined in this
       * namespace
       *
       * @ingroup EulerEquations
       */
      template <int dim, typename Number>
      static void populate_initial_state_list(
          initial_state_list_type<dim, Number> &initial_state_list,
          const HyperbolicSystem::View<dim, Number> &h,
          const std::string &s)
      {
      }
    };
  } // namespace Euler
} // namespace ryujin
