//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/euler_aeos/hyperbolic_system.h>
#include <ryujin/euler_aeos/indicator.h>
#include <ryujin/euler_aeos/limiter.h>
#include <ryujin/euler_aeos/wave_speed_estimator.h>

namespace ryujin
{
  namespace EulerAEOS
  {
    /**
     * A struct that contains all equation specific classes describing the
     * chosen hyperbolic system, the indicator, the limiter and
     * (approximate) Riemann solver.
     *
     * The compressible Euler equations of gas dynamics. Generalized
     * implementation with a modified approximative Riemann solver,
     * indicator, and limiter suitable for arbitrary equations of state.
     *
     * @ingroup EulerEquations
     */
    struct HyperbolicDescription {
      using HyperbolicSystem = EulerAEOS::HyperbolicSystem;

      template <typename ScalarNumber = double>
      using Indicator = EulerAEOS::Indicator<ScalarNumber>;

      template <typename ScalarNumber = double>
      using Limiter = EulerAEOS::Limiter<ScalarNumber>;

      template <typename ScalarNumber = double>
      using WaveSpeedEstimator = EulerAEOS::WaveSpeedEstimator<ScalarNumber>;
    };
  } // namespace EulerAEOS
} // namespace ryujin
