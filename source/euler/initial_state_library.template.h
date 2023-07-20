//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <initial_state_library.h>

#include "description.h"
#include "initial_state_becker_solution.h"
#include "initial_state_contrast.h"
#include "initial_state_four_state_contrast.h"
#include "initial_state_isentropic_vortex.h"
#include "initial_state_leblanc.h"
#include "initial_state_noh.h"
#include "initial_state_radial_contrast.h"
#include "initial_state_ramp_up.h"
#include "initial_state_rarefaction.h"
#include "initial_state_shock_front.h"
#include "initial_state_uniform.h"

namespace ryujin
{
  using namespace Euler; // FIXME
  using namespace EulerInitialStates;

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
                                const HyperbolicSystemView &h,
                                const std::string &s)
    {
      auto add = [&](auto &&object) {
        initial_state_list.emplace(std::move(object));
      };

      add(std::make_unique<BeckerSolution<Description, dim, Number>>(h, s));
      add(std::make_unique<Contrast<Description, dim, Number>>(h, s));
      add(std::make_unique<IsentropicVortex<Description, dim, Number>>(h, s));
      add(std::make_unique<LeBlanc<Description, dim, Number>>(h, s));
      add(std::make_unique<Noh<Description, dim, Number>>(h, s));
      add(std::make_unique<RadialContrast<Description, dim, Number>>(h, s));
      add(std::make_unique<RampUp<Description, dim, Number>>(h, s));
      add(std::make_unique<Rarefaction<Description, dim, Number>>(h, s));
      add(std::make_unique<ShockFront<Description, dim, Number>>(h, s));
      add(std::make_unique<FourStateContrast<Description, dim, Number>>(h, s));
      add(std::make_unique<Uniform<Description, dim, Number>>(h, s));
    }
  };
} // namespace ryujin

