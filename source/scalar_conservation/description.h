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
     * The parabolic subsystem is chosen to be the identity.
     *
     * @note We group all of these templates together in this struct so
     * that we only need to add a single template parameter to the all the
     * algorithm classes, such as HyperbolicModule.
     *
     * @ingroup ScalarConservationEquations
     */
    struct Description {
      using HyperbolicSystem = ScalarConservation::HyperbolicSystem;

      template <int dim, typename Number = double>
      using HyperbolicSystemView =
          ScalarConservation::HyperbolicSystemView<dim, Number>;

      using ParabolicSystem = ryujin::StubParabolicSystem;

      template <int dim, typename Number = double>
      using ParabolicModule =
          ryujin::StubParabolicModule<Description, dim, Number>;

      template <int dim, typename Number = double>
      using Indicator = ScalarConservation::IndicatorView<dim, Number>;

      template <int dim, typename Number = double>
      using Limiter = ScalarConservation::LimiterView<dim, Number>;

      template <int dim, typename Number = double>
      using WaveSpeedEstimator =
          ScalarConservation::WaveSpeedEstimatorView<dim, Number>;
    };
  } // namespace ScalarConservation
} // namespace ryujin
