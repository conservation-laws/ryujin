//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/euler/initial_state_astro_jet.h>
#include <ryujin/euler/initial_state_becker_solution.h>
#include <ryujin/euler/initial_state_contrast.h>
#include <ryujin/euler/initial_state_exact_riemann_solution.h>
#include <ryujin/euler/initial_state_four_state_contrast.h>
#include <ryujin/euler/initial_state_function.h>
#include <ryujin/euler/initial_state_icf_like.h>
#include <ryujin/euler/initial_state_isentropic_vortex.h>
#include <ryujin/euler/initial_state_leblanc.h>
#include <ryujin/euler/initial_state_noh.h>
#include <ryujin/euler/initial_state_radial_contrast.h>
#include <ryujin/euler/initial_state_ramp_up.h>
#include <ryujin/euler/initial_state_rarefaction.h>
#include <ryujin/euler/initial_state_shock_front.h>
#include <ryujin/euler/initial_state_smooth_wave.h>
#include <ryujin/euler/initial_state_three_state_contrast.h>
#include <ryujin/euler/initial_state_uniform.h>
#include <ryujin/typed/initial_state.h>

#include <vector>

namespace ryujin
{
  namespace EulerInitialStates
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
          L::template creator<AstroJet<HS, dim, Number>>(),
          L::template creator<BeckerSolution<HS, dim, Number>>(),
          L::template creator<Contrast<HS, dim, Number>>(),
          L::template creator<ExactRiemannSolution<HS, dim, Number>>(),
          L::template creator<FourStateContrast<HS, dim, Number>>(),
          L::template creator<Function<HS, dim, Number>>(),
          L::template creator<ICFLike<HS, dim, Number>>(),
          L::template creator<IsentropicVortex<HS, dim, Number>>(),
          L::template creator<LeBlanc<HS, dim, Number>>(),
          L::template creator<Noh<HS, dim, Number>>(),
          L::template creator<RadialContrast<HS, dim, Number>>(),
          L::template creator<RampUp<HS, dim, Number>>(),
          L::template creator<Rarefaction<HS, dim, Number>>(),
          L::template creator<ShockFront<HS, dim, Number>>(),
          L::template creator<SmoothWave<HS, dim, Number>>(),
          L::template creator<ThreeStateContrast<HS, dim, Number>>(),
          L::template creator<Uniform<HS, dim, Number>>(),
      };
    }
  } // namespace EulerInitialStates
} // namespace ryujin
