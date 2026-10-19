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
#include <ryujin/typed/initial_state_library.h>

namespace ryujin
{
  namespace EulerInitialStates
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
      add(std::make_unique<AstroJet<HD, dim, Number>>(h, s));
      add(std::make_unique<BeckerSolution<HD, dim, Number>>(h, s));
      add(std::make_unique<Contrast<HD, dim, Number>>(h, s));
      add(std::make_unique<ExactRiemannSolution<HD, dim, Number>>(h, s));
      add(std::make_unique<FourStateContrast<HD, dim, Number>>(h, s));
      add(std::make_unique<Function<HD, dim, Number>>(h, s));
      add(std::make_unique<ICFLike<HD, dim, Number>>(h, s));
      add(std::make_unique<IsentropicVortex<HD, dim, Number>>(h, s));
      add(std::make_unique<LeBlanc<HD, dim, Number>>(h, s));
      add(std::make_unique<Noh<HD, dim, Number>>(h, s));
      add(std::make_unique<RadialContrast<HD, dim, Number>>(h, s));
      add(std::make_unique<RampUp<HD, dim, Number>>(h, s));
      add(std::make_unique<Rarefaction<HD, dim, Number>>(h, s));
      add(std::make_unique<ShockFront<HD, dim, Number>>(h, s));
      add(std::make_unique<SmoothWave<HD, dim, Number>>(h, s));
      add(std::make_unique<ThreeStateContrast<HD, dim, Number>>(h, s));
      add(std::make_unique<Uniform<HD, dim, Number>>(h, s));
    }
  } // namespace EulerInitialStates
} // namespace ryujin
