//
// SPDX-License-Identifier: MIT or BSD-3-Clause
// [LANL Copyright Statement]
// Copyright (C) 2020 - 2023 by the ryujin authors
// Copyright (C) 2023 - 2023 by Triad National Security, LLC
//

#pragma once

#include "../stub_solver.h"
#include "hyperbolic_system.h"
#include "indicator.h"
#include "limiter.h"
#include "parabolic_system.h"
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

      using ParabolicSystem = ShallowWater::ParabolicSystem;

      template <int dim, typename Number = double>
      using ParabolicSolver = ryujin::StubSolver<Description, dim, Number>;

      template <int dim, typename Number = double>
      using Indicator = ShallowWater::Indicator<dim, Number>;

      template <int dim, typename Number = double>
      using Limiter = ShallowWater::Limiter<dim, Number>;

      template <int dim, typename Number = double>
      using RiemannSolver = ShallowWater::RiemannSolver<dim, Number>;
    };
  } // namespace ShallowWater
} // namespace ryujin
