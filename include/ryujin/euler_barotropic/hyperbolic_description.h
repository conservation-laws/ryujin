//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/euler_barotropic/hyperbolic_system.h>
#include <ryujin/euler_barotropic/indicator.h>
#include <ryujin/euler_barotropic/limiter.h>
#include <ryujin/euler_barotropic/wave_speed_estimator.h>
#include <ryujin/typed/hyperbolic_module.h>

namespace ryujin
{
  namespace EulerBarotropic
  {
    /**
     * A struct that contains all equation specific classes describing the
     * chosen hyperbolic system, the indicator, the limiter and
     * (approximate) Riemann solver.
     *
     * The compressible Euler equations of gas dynamics. Specialized
     * implementation for a subclass of barotropic equations of state where
     * the pressure, internal energy and entropies are a function of the
     * density. We use a specialied Riemann solver, entropy viscosity
     * commutator, and limiter for this class of equations.
     *
     * @ingroup EulerBarotropicEquations
     */
    struct HyperbolicDescription {
      using HyperbolicSystem = EulerBarotropic::HyperbolicSystem;

      template <typename ScalarNumber = double>
      using Indicator = EulerBarotropic::Indicator<ScalarNumber>;

      template <typename ScalarNumber = double>
      using Limiter = EulerBarotropic::Limiter<ScalarNumber>;

      template <typename ScalarNumber = double>
      using WaveSpeedEstimator =
          EulerBarotropic::WaveSpeedEstimator<ScalarNumber>;

      template <int dim, typename Number>
      using HyperbolicModule =
          ryujin::HyperbolicModule<HyperbolicDescription, dim, Number>;
    };
  } // namespace EulerBarotropic
} // namespace ryujin
