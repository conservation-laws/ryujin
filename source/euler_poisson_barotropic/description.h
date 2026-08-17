//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include "../euler_barotropic/hyperbolic_system.h"
#include "../euler_barotropic/indicator.h"
#include "../euler_barotropic/limiter.h"
#include "../euler_barotropic/wave_speed_estimator.h"
#include "../euler_poisson/parabolic_module.h"
#include "../euler_poisson/parabolic_system.h"

namespace ryujin
{
  namespace EulerPoissonBarotropic
  {
    struct Description {
      using HyperbolicSystem = EulerBarotropic::HyperbolicSystem;

      template <int dim, typename Number = double>
      using HyperbolicSystemView =
          EulerBarotropic::HyperbolicSystemView<dim, Number>;

      using ParabolicSystem = EulerPoisson::ParabolicSystem;

      template <int dim, typename Number = double>
      using ParabolicModule =
          EulerPoisson::ParabolicModule<Description, dim, Number>;

      template <int dim, typename Number = double>
      using IndicatorView = EulerBarotropic::IndicatorView<dim, Number>;

      template <int dim, typename Number = double>
      using LimiterView = EulerBarotropic::LimiterView<dim, Number>;

      template <int dim, typename Number = double>
      using WaveSpeedEstimatorView =
          EulerBarotropic::WaveSpeedEstimatorView<dim, Number>;
    };
  } // namespace EulerPoissonBarotropic
} // namespace ryujin
