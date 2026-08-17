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
     * The parabolic subsystem is chosen to be the identity.
     *
     * @ingroup EulerEquations
     */
    struct Description {
      using HyperbolicSystem = EulerBarotropic::HyperbolicSystem;

      template <int dim, typename Number = double>
      using HyperbolicSystemView =
          EulerBarotropic::HyperbolicSystemView<dim, Number>;

      using ParabolicSystem = ryujin::StubParabolicSystem;

      template <int dim, typename Number = double>
      using ParabolicModule =
          ryujin::StubParabolicModule<Description, dim, Number>;

      template <int dim, typename Number = double>
      using Indicator = EulerBarotropic::Indicator<dim, Number>;

      template <int dim, typename Number = double>
      using Limiter = EulerBarotropic::Limiter<dim, Number>;

      template <int dim, typename Number = double>
      using WaveSpeedEstimator =
          EulerBarotropic::WaveSpeedEstimator<dim, Number>;
    };
  } // namespace EulerBarotropic
} // namespace ryujin
