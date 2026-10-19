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
    template <typename HyperbolicDescription, int dim, typename Number>
    auto defaults()
    {
      using HD = HyperbolicDescription;
      using L = InitialStateLibrary<HD, dim, Number>;
      return std::vector<typename L::Creator>{
          L::template creator<AstroJet<HD, dim, Number>>(),
          L::template creator<BeckerSolution<HD, dim, Number>>(),
          L::template creator<Contrast<HD, dim, Number>>(),
          L::template creator<ExactRiemannSolution<HD, dim, Number>>(),
          L::template creator<FourStateContrast<HD, dim, Number>>(),
          L::template creator<Function<HD, dim, Number>>(),
          L::template creator<ICFLike<HD, dim, Number>>(),
          L::template creator<IsentropicVortex<HD, dim, Number>>(),
          L::template creator<LeBlanc<HD, dim, Number>>(),
          L::template creator<Noh<HD, dim, Number>>(),
          L::template creator<RadialContrast<HD, dim, Number>>(),
          L::template creator<RampUp<HD, dim, Number>>(),
          L::template creator<Rarefaction<HD, dim, Number>>(),
          L::template creator<ShockFront<HD, dim, Number>>(),
          L::template creator<SmoothWave<HD, dim, Number>>(),
          L::template creator<ThreeStateContrast<HD, dim, Number>>(),
          L::template creator<Uniform<HD, dim, Number>>(),
      };
    }
  } // namespace EulerInitialStates
} // namespace ryujin
