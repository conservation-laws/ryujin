//
// SPDX-License-Identifier: Apache-2.0
// [LANL Copyright Statement]
// Copyright (C) 2023 - 2025 by the ryujin authors
// Copyright (C) 2023 - 2024 by Triad National Security, LLC
//

#pragma once

#include <compile_time_options.h>

#include "../stub_parabolic_module.h"
#include "../stub_parabolic_system.h"
#include "hyperbolic_system.h"
#include "indicator.h"
#include "limiter.h"
#include "wave_speed_estimator.h"

namespace ryujin
{
  namespace ShallowWater
  {
    /**
     * A struct that contains all equation specific classes describing the
     * chosen hyperbolic system, the indicator, the limiter and
     * (approximate) Riemann solver.
     *
     * The compressible shallow water equations.
     *
     * The parabolic subsystem is chosen to be the identity.
     *
     * @ingroup ShallowWaterEquations
     */
    struct Description {
      using HyperbolicSystem = ShallowWater::HyperbolicSystem;

      using ParabolicSystem = ryujin::StubParabolicSystem;

      template <int dim, typename Number = double>
      using ParabolicModule =
          ryujin::StubParabolicModule<Description, dim, Number>;

      template <typename ScalarNumber = double>
      using Indicator = ShallowWater::Indicator<ScalarNumber>;

      template <typename ScalarNumber = double>
      using Limiter = ShallowWater::Limiter<ScalarNumber>;

      template <typename ScalarNumber = double>
      using WaveSpeedEstimator = ShallowWater::WaveSpeedEstimator<ScalarNumber>;
    };
  } // namespace ShallowWater
} // namespace ryujin
