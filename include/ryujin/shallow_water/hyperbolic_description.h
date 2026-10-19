//
// SPDX-License-Identifier: Apache-2.0
// [LANL Copyright Statement]
// Copyright (C) 2023 - 2026 by the ryujin authors
// Copyright (C) 2023 - 2024 by Triad National Security, LLC
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/shallow_water/hyperbolic_system.h>
#include <ryujin/shallow_water/indicator.h>
#include <ryujin/shallow_water/limiter.h>
#include <ryujin/shallow_water/wave_speed_estimator.h>

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
     * @ingroup ShallowWaterEquations
     */
    struct HyperbolicDescription {
      using HyperbolicSystem = ShallowWater::HyperbolicSystem;

      template <typename ScalarNumber = double>
      using Indicator = ShallowWater::Indicator<ScalarNumber>;

      template <typename ScalarNumber = double>
      using Limiter = ShallowWater::Limiter<ScalarNumber>;

      template <typename ScalarNumber = double>
      using WaveSpeedEstimator = ShallowWater::WaveSpeedEstimator<ScalarNumber>;
    };
  } // namespace ShallowWater
} // namespace ryujin
