//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 by the ryujin authors
// Copyright (C) 2025 by Triad National Security, LLC
//

#pragma once

#include "hyperbolic_system.h"

#include <initial_state_library.h>

#include "initial_state_contrast.h"
#include "initial_state_exact_riemann_solution.h"
#include "initial_state_function.h"
#include "initial_state_icf_like.h"
#include "initial_state_radial_contrast.h"
#include "initial_state_shock_bubble.h"
#include "initial_state_smooth_wave.h"
#include "initial_state_three_state_contrast.h"


namespace ryujin
{
  namespace MultiSpeciesEulerInitialStates
  {
    using namespace MultiSpeciesEuler;

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

      /* ExactRiemannSolution only supports 2 species */
      if constexpr (n_species == 2)
        add(std::make_unique<ExactRiemannSolution<Description, dim, Number>>(h,
                                                                             s));

      add(std::make_unique<Function<Description, dim, Number>>(h, s));
      add(std::make_unique<ICFLike<Description, dim, Number>>(h, s));
      add(std::make_unique<RadialContrast<Description, dim, Number>>(h, s));
      add(std::make_unique<ShockBubble<Description, dim, Number>>(h, s));
      add(std::make_unique<SmoothWave<Description, dim, Number>>(h, s));
      add(std::make_unique<ThreeStateContrast<Description, dim, Number>>(h, s));
    }
  } // namespace MultiSpeciesEulerInitialStates
} // namespace ryujin
