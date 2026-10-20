//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// [LANL Copyright Statement]
// Copyright (C) 2023 - 2026 by the ryujin authors
// Copyright (C) 2025 - 2026 by Triad National Security, LLC
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/euler_atmospheric/hyperbolic_system.h>
#include <ryujin/euler_atmospheric/indicator.h>
#include <ryujin/euler_atmospheric/limiter.h>
#include <ryujin/euler_atmospheric/wave_speed_estimator.h>
#include <ryujin/typed/hyperbolic_module.h>

namespace ryujin
{
  namespace EulerAtmospheric
  {
    /**
     * A struct that contains all equation specific classes describing the
     * chosen hyperbolic system, the indicator, the limiter and
     * (approximate) Riemann solver.
     *
     * The compressible atmospheric Euler equations with an ideal gas
     * equation of state, formulated in terms of the potential temperature
     * instead of the total energy.
     *
     * @ingroup EulerAtmosphericEquations
     */
    struct HyperbolicDescription {
      using HyperbolicSystem = EulerAtmospheric::HyperbolicSystem;

      template <int dim, typename Number>
      using InitialValues =
          ryujin::InitialValues<HyperbolicSystem, dim, Number>;

      template <typename ScalarNumber = double>
      using Indicator = EulerAtmospheric::Indicator<ScalarNumber>;

      template <typename ScalarNumber = double>
      using Limiter = EulerAtmospheric::Limiter<ScalarNumber>;

      template <typename ScalarNumber = double>
      using WaveSpeedEstimator =
          EulerAtmospheric::WaveSpeedEstimator<ScalarNumber>;

      template <int dim, typename Number>
      using HyperbolicModule =
          ryujin::HyperbolicModule<HyperbolicDescription, dim, Number>;
    };
  } // namespace EulerAtmospheric
} // namespace ryujin
