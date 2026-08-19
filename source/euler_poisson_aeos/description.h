//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include "../euler_aeos/hyperbolic_system.h"
#include "../euler_aeos/indicator.h"
#include "../euler_aeos/limiter.h"
#include "../euler_aeos/wave_speed_estimator.h"
#include "../euler_poisson/parabolic_module.h"
#include "../euler_poisson/parabolic_system.h"

namespace ryujin
{
  namespace EulerPoissonAEOS
  {
    struct Description {
      using HyperbolicSystem = EulerAEOS::HyperbolicSystem;

      using ParabolicSystem = EulerPoisson::ParabolicSystem;

      template <int dim, typename Number = double>
      using ParabolicModule =
          EulerPoisson::ParabolicModule<Description, dim, Number>;

      template <typename ScalarNumber = double>
      using Indicator = EulerAEOS::Indicator<ScalarNumber>;

      template <typename ScalarNumber = double>
      using Limiter = EulerAEOS::Limiter<ScalarNumber>;

      template <typename ScalarNumber = double>
      using WaveSpeedEstimator = EulerAEOS::WaveSpeedEstimator<ScalarNumber>;
    };
  } // namespace EulerPoissonAEOS
} // namespace ryujin
