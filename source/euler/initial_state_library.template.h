//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "description.h"
#include "initial_state_library.h"

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

namespace ryujin
{
  namespace Euler
  {
    template <int dim, typename Number>
    void InitialStateLibrary::populate_initial_state_list(
        initial_state_list_type<dim, Number> &initial_state_list,
        const HyperbolicSystem::View<dim, Number> &h,
        const std::string &s)
    {
      auto add = [&](auto &&object) {
        initial_state_list.emplace(std::move(object));
      };

      add(std::make_unique<BeckerSolution<dim, Number>>(h, s));
      add(std::make_unique<Contrast<dim, Number>>(h, s));
      add(std::make_unique<IsentropicVortex<dim, Number>>(h, s));
      add(std::make_unique<LeBlanc<dim, Number>>(h, s));
      add(std::make_unique<Noh<dim, Number>>(h, s));
      add(std::make_unique<RadialContrast<dim, Number>>(h, s));
      add(std::make_unique<RampUp<dim, Number>>(h, s));
      add(std::make_unique<Rarefaction<dim, Number>>(h, s));
      add(std::make_unique<ShockFront<dim, Number>>(h, s));
      add(std::make_unique<FourStateContrast<dim, Number>>(h, s));
      add(std::make_unique<Uniform<dim, Number>>(h, s));
    }
  } // namespace Euler
} // namespace ryujin
