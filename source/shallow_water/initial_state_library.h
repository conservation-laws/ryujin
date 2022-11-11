//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <initial_state.h>

#include <hyperbolic_system.h>

#include "initial_state_circular_dam_break.h"
#include "initial_state_flow_over_bump.h"
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
  namespace ShallowWater
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
                                       const HyperbolicSystem &h,
                                       const std::string &s)
      {
        using state_type = HyperbolicSystem::state_type<dim, Number>;

        auto add = [&](auto &&object) {
          initial_state_list.emplace(std::move(object));
        };

        add(std::make_unique<CircularDamBreak<dim, Number, state_type>>(h, s));
        add(std::make_unique<FlowOverBump<dim, Number, state_type>>(h, s));
        add(std::make_unique<Paraboloid<dim, Number, state_type>>(h, s));
        add(std::make_unique<RitterDamBreak<dim, Number, state_type>>(h, s));
        add(std::make_unique<SolitaryWave<dim, Number, state_type>>(h, s));
        add(std::make_unique<ThreeBumpsDamBreak<dim, Number, state_type>>(h,
                                                                          s));
        add(std::make_unique<TriangularDamBreak<dim, Number, state_type>>(h,
                                                                          s));
        add(std::make_unique<Uniform<dim, Number, state_type>>(h, s));
        add(std::make_unique<SlopingRampDamBreak<dim, Number, state_type>>(h,
                                                                           s));
        add(std::make_unique<UnsteadyVortex<dim, Number, state_type>>(h, s));
      }

    } // namespace InitialStateLibrary
  } // namespace ShallowWater
} // namespace ryujin
