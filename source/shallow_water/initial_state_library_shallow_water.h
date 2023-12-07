//
// SPDX-License-Identifier: MIT or BSD-3-Clause
// [LANL Copyright Statement]
// Copyright (C) 2020 - 2023 by the ryujin authors
// Copyright (C) 2023 - 2023 by Triad National Security, LLC
//

#pragma once

#include <initial_state_library.h>

#include "initial_state_contrast.h"
#include "initial_state_flow_over_bump.h"
#include "initial_state_function.h"
#include "initial_state_hou_test.h"
#include "initial_state_paraboloid.h"
#include "initial_state_ritter_dam_break.h"
#include "initial_state_smooth_vortex.h"
#include "initial_state_three_bumps_dam_break.h"
#include "initial_state_uniform.h"


namespace ryujin
{
  namespace ShallowWaterInitialStates
  {
    template <typename Description, int dim, typename Number>
    void populate_initial_state_list(
        typename ryujin::InitialStateLibrary<Description, dim, Number>::
            initial_state_list_type &initial_state_list,
        const typename Description::HyperbolicSystem &h,
        const std::string &s)
    {
      auto add = [&](auto &&object) {
        initial_state_list.emplace(std::move(object));
      };

      add(std::make_unique<Contrast<Description, dim, Number>>(h, s));
      add(std::make_unique<FlowOverBump<Description, dim, Number>>(h, s));
      add(std::make_unique<Function<Description, dim, Number>>(h, s));
      add(std::make_unique<HouTest<Description, dim, Number>>(h, s));
      add(std::make_unique<Paraboloid<Description, dim, Number>>(h, s));
      add(std::make_unique<RitterDamBreak<Description, dim, Number>>(h, s));
      add(std::make_unique<SmoothVortex<Description, dim, Number>>(h, s));
      add(std::make_unique<ThreeBumpsDamBreak<Description, dim, Number>>(h, s));
      add(std::make_unique<Uniform<Description, dim, Number>>(h, s));
    }
  } // namespace ShallowWaterInitialStates
} // namespace ryujin
