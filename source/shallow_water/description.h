//
// SPDX-License-Identifier: Apache-2.0
// [LANL Copyright Statement]
// Copyright (C) 2023 - 2025 by the ryujin authors
// Copyright (C) 2023 - 2024 by Triad National Security, LLC
//

#pragma once

#include <compile_time_options.h>

#include "../stub_parabolic_module.h"
#include "../stub_parabolic_system.h"
#include "hyperbolic_system.h"
#include "indicator.h"
#include "limiter.h"
#include "riemann_solver.h"

namespace ryujin
{
  namespace ShallowWater
  {
    /**
     * A struct that contains all equation specific classes describing the
     * chosen hyperbolic system, the indicator, the limiter and
     * (approximate) Riemann solver.
     *
     * The compressible shallow water equations.
     *
     * The parabolic subsystem is chosen to be the identity.
     *
     * @ingroup ShallowWaterEquations
     */
    struct Description {
      using HyperbolicSystem = ShallowWater::HyperbolicSystem;

      template <int dim, typename Number = double>
      using HyperbolicSystemView =
          ShallowWater::HyperbolicSystemView<dim, Number>;

      using ParabolicSystem = ryujin::StubParabolicSystem;

      template <int dim, typename Number = double>
      using ParabolicModule =
          ryujin::StubParabolicModule<Description, dim, Number>;

      template <int dim, typename Number = double>
      using Indicator = ShallowWater::Indicator<dim, Number>;

      template <int dim, typename Number = double>
      using Limiter = ShallowWater::Limiter<dim, Number>;

      template <int dim, typename Number = double>
      using RiemannSolver = ShallowWater::RiemannSolver<dim, Number>;
    };
  } // namespace ShallowWater
} // namespace ryujin
