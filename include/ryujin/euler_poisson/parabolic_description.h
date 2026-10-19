//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/euler_poisson/parabolic_module.h>
#include <ryujin/euler_poisson/parabolic_system.h>

namespace ryujin
{
  namespace EulerPoisson
  {
    /**
     * A struct that contains the parabolic system and the parabolic module
     * of an electrostatic force model with a background magnetic field.
     * The electrostatic potential is obtained by solving a Poisson problem
     * for the given density distribution.
     *
     * The parabolic module can be paired with the hyperbolic descriptions
     * of the compressible Euler equations: Euler::HyperbolicDescription,
     * EulerAEOS::HyperbolicDescription, and
     * EulerBarotropic::HyperbolicDescription.
     *
     * @ingroup EulerPoissonEquations
     */
    struct ParabolicDescription {
      using ParabolicSystem = EulerPoisson::ParabolicSystem;

      template <typename HyperbolicDescription, int dim, typename Number>
      using ParabolicModule =
          EulerPoisson::ParabolicModule<HyperbolicDescription, dim, Number>;
    };
  } // namespace EulerPoisson
} // namespace ryujin
