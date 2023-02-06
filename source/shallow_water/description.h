//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

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
     * We group all of these templates together in this struct so that we
     * only need to add a single template parameter to the all the
     * algorithm classes, such as HyperbolicModule.
     *
     * @ingroup EulerEquations
     */
    struct Description {
      using HyperbolicSystem = ShallowWater::HyperbolicSystem;

      template <int dim, typename Number = double>
      using Indicator = ShallowWater::Indicator<dim, Number>;

      template <int dim, typename Number = double>
      using Limiter = ShallowWater::Limiter<dim, Number>;

      template <int dim, typename Number = double>
      using RiemannSolver = ShallowWater::RiemannSolver<dim, Number>;
    };
  } // namespace Euler
} // namespace ryujin
