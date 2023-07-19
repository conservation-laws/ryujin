//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"
#include <initial_state.h>

#include "initial_state_uniform.h"

namespace ryujin
{
  namespace Skeleton
  {
    struct InitialStateLibrary {
      /**
       * Populate a given container with all initial states defined in this
       * namespace
       *
       * @ingroup SkeletonEquations
       */
      template <int dim, typename Number, typename T>
      static void
      populate_initial_state_list(T &initial_state_list,
                                  const HyperbolicSystem::View<dim, Number> &h,
                                  const std::string &s)
      {
        auto add = [&](auto &&object) {
          initial_state_list.emplace(std::move(object));
        };

        add(std::make_unique<Uniform<dim, Number>>(h, s));
      }
    };
  } // namespace Skeleton
} // namespace ryujin
