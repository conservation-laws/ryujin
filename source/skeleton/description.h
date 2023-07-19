//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"
#include "indicator.h"
#include "initial_state_library.h"
#include "limiter.h"
#include "riemann_solver.h"

namespace ryujin
{
  namespace Skeleton
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
     * @ingroup SkeletonEquations
     */
    struct Description {
      using HyperbolicSystem = Skeleton::HyperbolicSystem;

      template <int dim, typename Number = double>
      using Indicator = Skeleton::Indicator<dim, Number>;

      using InitialStateLibrary = Skeleton::InitialStateLibrary;

      template <int dim, typename Number = double>
      using Limiter = Skeleton::Limiter<dim, Number>;

      template <int dim, typename Number = double>
      using RiemannSolver = Skeleton::RiemannSolver<dim, Number>;
    };
  } // namespace Skeleton
} // namespace ryujin
