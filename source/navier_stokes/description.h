//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "../euler/hyperbolic_system.h"
#include "../euler/indicator.h"
#include "../euler/limiter.h"
#include "../euler/wave_speed_estimator.h"
#include "parabolic_module.h"
#include "parabolic_system.h"

namespace ryujin
{
  namespace NavierStokes
  {
    /**
     * A struct that contains all equation specific classes describing the
     * chosen hyperbolic system, the indicator, the limiter and
     * (approximate) Riemann solver.
     *
     * We group all of these templates together in this struct so that we
     * only need to add a single template parameter to the all the
     * algorithm classes, such as HyperbolicModule.
     *
     * @ingroup NavierStokesEquations
     */
    struct Description {
      using HyperbolicSystem = Euler::HyperbolicSystem;

      template <int dim, typename Number = double>
      using HyperbolicSystemView = Euler::HyperbolicSystemView<dim, Number>;

      using ParabolicSystem = NavierStokes::ParabolicSystem;

      template <int dim, typename Number = double>
      using ParabolicModule = NavierStokes::ParabolicModule<dim, Number>;

      template <int dim, typename Number = double>
      using Indicator = Euler::IndicatorView<dim, Number>;

      template <int dim, typename Number = double>
      using Limiter = Euler::LimiterView<dim, Number>;

      template <int dim, typename Number = double>
      using WaveSpeedEstimator = Euler::WaveSpeedEstimatorView<dim, Number>;
    };
  } // namespace NavierStokes
} // namespace ryujin
