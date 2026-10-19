//
// SPDX-License-Identifier: Apache-2.0
// [LANL Copyright Statement]
// Copyright (C) 2023 - 2026 by the ryujin authors
// Copyright (C) 2023 - 2024 by Triad National Security, LLC
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/shallow_water/initial_state_contrast.h>
#include <ryujin/shallow_water/initial_state_flow_over_bump.h>
#include <ryujin/shallow_water/initial_state_function.h>
#include <ryujin/shallow_water/initial_state_geotiff.h>
#include <ryujin/shallow_water/initial_state_hou_test.h>
#include <ryujin/shallow_water/initial_state_paraboloid.h>
#include <ryujin/shallow_water/initial_state_ritter_dam_break.h>
#include <ryujin/shallow_water/initial_state_sloping_friction.h>
#include <ryujin/shallow_water/initial_state_smooth_vortex.h>
#include <ryujin/shallow_water/initial_state_soliton.h>
#include <ryujin/shallow_water/initial_state_three_bumps_dam_break.h>
#include <ryujin/shallow_water/initial_state_transient.h>
#include <ryujin/shallow_water/initial_state_uniform.h>
#include <ryujin/typed/initial_state_library.h>


namespace ryujin
{
  namespace ShallowWaterInitialStates
  {
    template <typename HyperbolicDescription, int dim, typename Number>
    void populate_initial_state_list(
        typename InitialStateLibrary<HyperbolicDescription, dim, Number>::
            initial_state_list_type &initial_state_list,
        const typename HyperbolicDescription::HyperbolicSystem &h,
        const std::string &s)
    {
      using HD = HyperbolicDescription;
      auto add = [&](auto &&object) {
        initial_state_list.emplace(std::move(object));
      };
      add(std::make_unique<Contrast<HD, dim, Number>>(h, s));
      add(std::make_unique<FlowOverBump<HD, dim, Number>>(h, s));
      add(std::make_unique<Function<HD, dim, Number>>(h, s));
      add(std::make_unique<GeoTIFF<HD, dim, Number>>(h, s));
      add(std::make_unique<HouTest<HD, dim, Number>>(h, s));
      add(std::make_unique<Paraboloid<HD, dim, Number>>(h, s));
      add(std::make_unique<RitterDamBreak<HD, dim, Number>>(h, s));
      add(std::make_unique<SlopingFriction<HD, dim, Number>>(h, s));
      add(std::make_unique<SmoothVortex<HD, dim, Number>>(h, s));
      add(std::make_unique<ThreeBumpsDamBreak<HD, dim, Number>>(h, s));
      add(std::make_unique<Uniform<HD, dim, Number>>(h, s));
      add(std::make_unique<TankExperiments<HD, dim, Number>>(h, s));
      add(std::make_unique<Soliton<HD, dim, Number>>(h, s));
    }
  } // namespace ShallowWaterInitialStates
} // namespace ryujin
