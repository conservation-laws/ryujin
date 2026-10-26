//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include <ryujin/typed/initial_state.h>

#include <vector>


namespace ryujin
{
  namespace ShallowWaterInitialStates
  {
    /**
     * The built-in initial states.
     */
    template <typename HyperbolicSystem, int dim, typename Number>
    auto defaults()
    {
      using HS = HyperbolicSystem;
      using L = InitialStateLibrary<HS, dim, Number>;
      return std::vector<typename L::Creator>{
          L::template creator<Contrast<HS, dim, Number>>(),
          L::template creator<FlowOverBump<HS, dim, Number>>(),
          L::template creator<Function<HS, dim, Number>>(),
          L::template creator<GeoTIFF<HS, dim, Number>>(),
          L::template creator<HouTest<HS, dim, Number>>(),
          L::template creator<Paraboloid<HS, dim, Number>>(),
          L::template creator<RitterDamBreak<HS, dim, Number>>(),
          L::template creator<SlopingFriction<HS, dim, Number>>(),
          L::template creator<SmoothVortex<HS, dim, Number>>(),
          L::template creator<ThreeBumpsDamBreak<HS, dim, Number>>(),
          L::template creator<Uniform<HS, dim, Number>>(),
          L::template creator<TankExperiments<HS, dim, Number>>(),
          L::template creator<Soliton<HS, dim, Number>>(),
      };
    }
  } // namespace ShallowWaterInitialStates
} // namespace ryujin
