//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2025 by the ryujin authors
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
     * The parabolic subsystem is chosen to be the identity.
     *
     * @ingroup EulerEquations
     */
    struct Description {
      using HyperbolicSystem = EulerAEOS::HyperbolicSystem;

      using ParabolicSystem = ryujin::StubParabolicSystem;

      template <int dim, typename Number = double>
      using ParabolicModule =
          ryujin::StubParabolicModule<Description, dim, Number>;

      template <typename ScalarNumber = double>
      using Indicator = EulerAEOS::Indicator<ScalarNumber>;

      template <typename ScalarNumber = double>
      using Limiter = EulerAEOS::Limiter<ScalarNumber>;

      template <typename ScalarNumber = double>
      using WaveSpeedEstimator = EulerAEOS::WaveSpeedEstimator<ScalarNumber>;
    };
  } // namespace EulerAEOS
} // namespace ryujin
