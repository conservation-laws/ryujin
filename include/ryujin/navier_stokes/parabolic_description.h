//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/navier_stokes/parabolic_module.h>
#include <ryujin/navier_stokes/parabolic_system.h>

namespace ryujin
{
  namespace NavierStokes
  {
    /**
     * A struct that contains the parabolic system and the parabolic module
     * of the compressible Navier-Stokes equations: a Newtonian fluid
     * viscosity model with a heat flux governed by Fourier's law.
     *
     * The parabolic module can only be paired with the hyperbolic
     * description of the compressible Euler equations,
     * Euler::HyperbolicDescription.
     *
     * @ingroup NavierStokesEquations
     */
    struct ParabolicDescription {
      using ParabolicSystem = NavierStokes::ParabolicSystem;

      template <typename HyperbolicDescription, int dim, typename Number>
      using ParabolicModule = NavierStokes::ParabolicModule<dim, Number>;
    };
  } // namespace NavierStokes
} // namespace ryujin
