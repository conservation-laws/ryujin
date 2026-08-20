//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include "../euler_barotropic/hyperbolic_system.h"
#include "../euler_barotropic/indicator.h"
#include "../euler_barotropic/limiter.h"
#include "../euler_barotropic/wave_speed_estimator.h"
#include "../euler_poisson/parabolic_module.h"
#include "../euler_poisson/parabolic_system.h"

namespace ryujin
{
  namespace EulerPoissonBarotropic
  {
    /**
     * A struct that contains all equation specific classes describing the
     * chosen hyperbolic system, the indicator, the limiter and
     * (approximate) Riemann solver.
     *
     * The compressible Euler equations of gas dynamics for a
     * barotropic equation of state, coupled to a Poisson problem for the
     * electrostatic potential.
     *
     * @ingroup EulerPoissonEquations
     */
    struct Description {
      using HyperbolicSystem = EulerBarotropic::HyperbolicSystem;

      using ParabolicSystem = EulerPoisson::ParabolicSystem;

      template <int dim, typename Number = double>
      using ParabolicModule =
          EulerPoisson::ParabolicModule<Description, dim, Number>;

      template <typename ScalarNumber = double>
      using Indicator = EulerBarotropic::Indicator<ScalarNumber>;

      template <typename ScalarNumber = double>
      using Limiter = EulerBarotropic::Limiter<ScalarNumber>;

      template <typename ScalarNumber = double>
      using WaveSpeedEstimator =
          EulerBarotropic::WaveSpeedEstimator<ScalarNumber>;
    };
  } // namespace EulerPoissonBarotropic
} // namespace ryujin
