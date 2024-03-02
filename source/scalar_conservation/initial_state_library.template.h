//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 by the ryujin authors
//

#pragma once

#include <initial_state_library.h>

#include "description.h"
#include "initial_state_function.h"
#include "initial_state_uniform.h"

namespace ryujin
{
  using namespace ScalarConservation;

  template <int dim, typename Number>
  class InitialStateLibrary<Description, dim, Number>
  {
  public:
    using HyperbolicSystem = typename Description::HyperbolicSystem;

    using initial_state_list_type =
        std::set<std::unique_ptr<InitialState<Description, dim, Number>>>;

    static void
    populate_initial_state_list(initial_state_list_type &initial_state_list,
                                const HyperbolicSystem &h,
                                const std::string &s)
    {
      auto add = [&](auto &&object) {
        initial_state_list.emplace(std::move(object));
      };

      add(std::make_unique<Function<dim, Number>>(h, s));
      add(std::make_unique<Uniform<dim, Number>>(h, s));
    }
  };
} // namespace ryujin
