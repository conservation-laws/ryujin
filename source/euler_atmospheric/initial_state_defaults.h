//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// [LANL Copyright Statement]
// Copyright (C) 2023 - 2026 by the ryujin authors
// Copyright (C) 2025 - 2026 by Triad National Security, LLC
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/euler_atmospheric/initial_state_contrast.h>
#include <ryujin/euler_atmospheric/initial_state_exact_riemann_solution.h>
#include <ryujin/euler_atmospheric/initial_state_exact_source.h>
#include <ryujin/euler_atmospheric/initial_state_function.h>
#include <ryujin/euler_atmospheric/initial_state_smooth_wave.h>
#include <ryujin/euler_atmospheric/initial_state_uniform.h>
#include <ryujin/typed/initial_state.h>

#include <vector>

namespace ryujin
{
  namespace EulerAtmosphericInitialStates
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
          L::template creator<ExactRiemannSolution<HS, dim, Number>>(),
          L::template creator<ExactSource<HS, dim, Number>>(),
          L::template creator<Function<HS, dim, Number>>(),
          L::template creator<SmoothWave<HS, dim, Number>>(),
          L::template creator<Uniform<HS, dim, Number>>(),
      };
    }
  } // namespace EulerAtmosphericInitialStates
} // namespace ryujin
