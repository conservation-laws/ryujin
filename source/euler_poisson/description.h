//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include "../euler/hyperbolic_system.h"
#include "../euler/indicator.h"
#include "../euler/limiter.h"
#include "../euler/wave_speed_estimator.h"
#include "parabolic_module.h"
#include "parabolic_system.h"

namespace ryujin
{
  namespace EulerPoisson
  {
    struct Description {
      using HyperbolicSystem = Euler::HyperbolicSystem;

      using ParabolicSystem = EulerPoisson::ParabolicSystem;

      template <int dim, typename Number = double>
      using ParabolicModule =
          EulerPoisson::ParabolicModule<Description, dim, Number>;

      template <typename ScalarNumber = double>
      using Indicator = Euler::Indicator<ScalarNumber>;

      template <typename ScalarNumber = double>
      using Limiter = Euler::Limiter<ScalarNumber>;

      template <typename ScalarNumber = double>
      using WaveSpeedEstimator = Euler::WaveSpeedEstimator<ScalarNumber>;
    };
  } // namespace EulerPoisson
} // namespace ryujin
