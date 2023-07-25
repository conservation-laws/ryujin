//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"
#include "indicator.h"
#include "limiter.h"
#include "riemann_solver.h"

namespace ryujin
{
  namespace EulerAEOS
  {
    /**
     * A struct that contains all equation specific classes describing the
     * chosen hyperbolic system, the indicator, the limiter and
     * (approximate) Riemann solver.
     *
     * The compressible Euler equations of gas dynamics. Generalized
     * implementation with a modified approximative Riemann solver,
     * indicator, and limiter suitable for arbitrary equations of state.
     *
     * The parabolic subsystem is chosen to be the identity.
     *
     * @ingroup EulerEquations
     */
    struct Description {
      using HyperbolicSystem = EulerAEOS::HyperbolicSystem;

      template <int dim, typename Number = double>
      using Indicator = EulerAEOS::Indicator<dim, Number>;

      template <int dim, typename Number = double>
      using Limiter = EulerAEOS::Limiter<dim, Number>;

      template <int dim, typename Number = double>
      using RiemannSolver = EulerAEOS::RiemannSolver<dim, Number>;
    };
  } // namespace EulerAEOS
} // namespace ryujin
