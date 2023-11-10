//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <initial_state_library.h>

#include "description.h"
#include "initial_state_circular_dam_break.h"
#include "initial_state_contrast.h"
#include "initial_state_flow_over_bump.h"
#include "initial_state_hou_test.h"
#include "initial_state_new_paraboloid.h"
#include "initial_state_paraboloid.h"
#include "initial_state_ritter_dam_break.h"
#include "initial_state_sloping_ramp_dam_break.h"
#include "initial_state_solitary_wave.h"
#include "initial_state_three_bumps_dam_break.h"
#include "initial_state_triangular_dam_break.h"
#include "initial_state_uniform.h"
#include "initial_state_unsteady_vortex.h"

namespace ryujin
{
  using namespace ShallowWater;

  template <int dim, typename Number>
  class InitialStateLibrary<Description, dim, Number>
  {
  public:
    using HyperbolicSystem = typename Description::HyperbolicSystem;

    using HyperbolicSystemView =
        typename HyperbolicSystem::template View<dim, Number>;

    using initial_state_list_type =
        std::set<std::unique_ptr<InitialState<Description, dim, Number>>>;

    /**
     * Populate a given container with all initial state defined in this
     * namespace
     *
     * @ingroup ShallowWaterEquations
     */
    static void
    populate_initial_state_list(initial_state_list_type &initial_state_list,
                                const HyperbolicSystem &h,
                                const std::string &s)
    {
      auto add = [&](auto &&object) {
        initial_state_list.emplace(std::move(object));
      };

      add(std::make_unique<CircularDamBreak<dim, Number>>(h, s));
      add(std::make_unique<Contrast<dim, Number>>(h, s));
      add(std::make_unique<FlowOverBump<dim, Number>>(h, s));
      add(std::make_unique<HouTest<dim, Number>>(h, s));
      add(std::make_unique<NewParaboloid<dim, Number>>(h, s));
      add(std::make_unique<Paraboloid<dim, Number>>(h, s));
      add(std::make_unique<RitterDamBreak<dim, Number>>(h, s));
      add(std::make_unique<SolitaryWave<dim, Number>>(h, s));
      add(std::make_unique<ThreeBumpsDamBreak<dim, Number>>(h, s));
      add(std::make_unique<TriangularDamBreak<dim, Number>>(h, s));
      add(std::make_unique<Uniform<dim, Number>>(h, s));
      add(std::make_unique<SlopingRampDamBreak<dim, Number>>(h, s));
      add(std::make_unique<UnsteadyVortex<dim, Number>>(h, s));
    }
  };
} // namespace ryujin
