//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/euler/hyperbolic_system.h>
#include <ryujin/euler/indicator.h>
#include <ryujin/euler/limiter.h>
#include <ryujin/euler/wave_speed_estimator.h>

namespace ryujin
{
  namespace Euler
  {
    /**
     * A struct that contains all equation specific classes describing the
     * chosen hyperbolic system, the indicator, the limiter and
     * (approximate) Riemann solver.
     *
     * The compressible Euler equations of gas dynamics. Specialized
     * implementation for a polytropic gas equation.
     *
     * @ingroup EulerEquations
     */
    struct HyperbolicDescription {
      using HyperbolicSystem = Euler::HyperbolicSystem;

      template <typename ScalarNumber = double>
      using Indicator = Euler::Indicator<ScalarNumber>;

      template <typename ScalarNumber = double>
      using Limiter = Euler::Limiter<ScalarNumber>;

      template <typename ScalarNumber = double>
      using WaveSpeedEstimator = Euler::WaveSpeedEstimator<ScalarNumber>;
    };
  } // namespace Euler
} // namespace ryujin
