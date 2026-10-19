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
#include <ryujin/typed/initial_state.h>

#include <vector>


namespace ryujin
{
  namespace ShallowWaterInitialStates
  {
    /**
     * The built-in initial states.
     */
    template <typename HyperbolicDescription, int dim, typename Number>
    auto defaults()
    {
      using HD = HyperbolicDescription;
      using L = InitialStateLibrary<HD, dim, Number>;
      return std::vector<typename L::Creator>{
          L::template creator<Contrast<HD, dim, Number>>(),
          L::template creator<FlowOverBump<HD, dim, Number>>(),
          L::template creator<Function<HD, dim, Number>>(),
          L::template creator<GeoTIFF<HD, dim, Number>>(),
          L::template creator<HouTest<HD, dim, Number>>(),
          L::template creator<Paraboloid<HD, dim, Number>>(),
          L::template creator<RitterDamBreak<HD, dim, Number>>(),
          L::template creator<SlopingFriction<HD, dim, Number>>(),
          L::template creator<SmoothVortex<HD, dim, Number>>(),
          L::template creator<ThreeBumpsDamBreak<HD, dim, Number>>(),
          L::template creator<Uniform<HD, dim, Number>>(),
          L::template creator<TankExperiments<HD, dim, Number>>(),
          L::template creator<Soliton<HD, dim, Number>>(),
      };
    }
  } // namespace ShallowWaterInitialStates
} // namespace ryujin
