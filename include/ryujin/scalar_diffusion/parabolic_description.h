//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/scalar_conservation/hyperbolic_description.h>
#include <ryujin/scalar_diffusion/parabolic_module.h>
#include <ryujin/scalar_diffusion/parabolic_system.h>

#include <concepts>

namespace ryujin
{
  namespace ScalarDiffusion
  {
    /**
     * A struct that contains the parabolic system and the parabolic module
     * of the power-law diffusion model.
     *
     * The parabolic module can only be paired with
     * ScalarConservation::HyperbolicDescription.
     *
     * @ingroup ScalarDiffusionEquations
     */
    struct ParabolicDescription {
      using ParabolicSystem = ScalarDiffusion::ParabolicSystem;

      template <typename HyperbolicDescription, int dim, typename Number>
        requires std::same_as<HyperbolicDescription,
                              ScalarConservation::HyperbolicDescription>
      using ParabolicModule = ScalarDiffusion::ParabolicModule<dim, Number>;
    };
  } // namespace ScalarDiffusion
} // namespace ryujin
