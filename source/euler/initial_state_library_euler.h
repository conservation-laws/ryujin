//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include <initial_state_library.h>

#include "initial_state_becker_solution.h"
#include "initial_state_contrast.h"
#include "initial_state_four_state_contrast.h"
#include "initial_state_isentropic_vortex.h"
#include "initial_state_leblanc.h"
#include "initial_state_noh.h"
#include "initial_state_radial_contrast.h"
#include "initial_state_ramp_up.h"
#include "initial_state_rarefaction.h"
#include "initial_state_shock_front.h"
#include "initial_state_uniform.h"
#include "initial_state_smooth_wave.h"
#include "initial_state_three_state_contrast.h"

namespace ryujin
{
  namespace EulerInitialStates
  {
    template <typename Description, int dim, typename Number>
    void populate_initial_state_list(
        typename ryujin::InitialStateLibrary<Description, dim, Number>::
            initial_state_list_type &initial_state_list,
        const typename Description::HyperbolicSystem::template View<dim, Number>
            &h,
        const std::string &s)
    {
      auto add = [&](auto &&object) {
        initial_state_list.emplace(std::move(object));
      };

      add(std::make_unique<BeckerSolution<Description, dim, Number>>(h, s));
      add(std::make_unique<Contrast<Description, dim, Number>>(h, s));
      add(std::make_unique<IsentropicVortex<Description, dim, Number>>(h, s));
      add(std::make_unique<LeBlanc<Description, dim, Number>>(h, s));
      add(std::make_unique<Noh<Description, dim, Number>>(h, s));
      add(std::make_unique<RadialContrast<Description, dim, Number>>(h, s));
      add(std::make_unique<RampUp<Description, dim, Number>>(h, s));
      add(std::make_unique<Rarefaction<Description, dim, Number>>(h, s));
      add(std::make_unique<ShockFront<Description, dim, Number>>(h, s));
      add(std::make_unique<FourStateContrast<Description, dim, Number>>(h, s));
      add(std::make_unique<Uniform<Description, dim, Number>>(h, s));
      add(std::make_unique<SmoothWave<Description, dim, Number>>(h, s));
      add(std::make_unique<ThreeStateContrast<Description, dim, Number>>(h, s));
    }
  } // namespace EulerInitialStates
} // namespace ryujin
