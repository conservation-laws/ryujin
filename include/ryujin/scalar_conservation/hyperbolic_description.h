//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "hyperbolic_system.h"
#include "indicator.h"
#include "limiter.h"
#include "wave_speed_estimator.h"

namespace ryujin
{
  namespace ScalarConservation
  {
    /**
     * This struct contains all equation specific classes describing the
     * chosen hyperbolic system, the indicator, the limiter and
     * (approximate) Riemann solver.
     *
     * A scalar conservation equation for a scalar unknown u with a
     * user-specified flux depending on the state u.
     *
     * @ingroup ScalarConservationEquations
     */
    struct HyperbolicDescription {
      using HyperbolicSystem = ScalarConservation::HyperbolicSystem;

      template <typename ScalarNumber = double>
      using Indicator = ScalarConservation::Indicator<ScalarNumber>;

      template <typename ScalarNumber = double>
      using Limiter = ScalarConservation::Limiter<ScalarNumber>;

      template <typename ScalarNumber = double>
      using WaveSpeedEstimator =
          ScalarConservation::WaveSpeedEstimator<ScalarNumber>;
    };
  } // namespace ScalarConservation
} // namespace ryujin
