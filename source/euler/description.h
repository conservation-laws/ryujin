//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
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
  namespace Euler
  {
    /**
     * A struct that contains all equation specific classes describing the
     * chosen hyperbolic system, the indicator, the limiter and
     * (approximate) Riemann solver.
     *
     * The compressible Euler equations of gas dynamics. Specialized
     * implementation for a polytropic gas equation.
     *
     * The parabolic subsystem is chosen to be the identity.
     *
     * @ingroup EulerEquations
     */
    struct Description {
      using HyperbolicSystem = Euler::HyperbolicSystem;

      using ParabolicSystem = Euler::ParabolicSystem;

      template <int dim, typename Number = double>
      using ParabolicSolver = ryujin::StubSolver<Description, dim, Number>;

      template <int dim, typename Number = double>
      using Indicator = Euler::Indicator<dim, Number>;

      template <int dim, typename Number = double>
      using Limiter = Euler::Limiter<dim, Number>;

      template <int dim, typename Number = double>
      using RiemannSolver = Euler::RiemannSolver<dim, Number>;
    };
  } // namespace Euler
} // namespace ryujin
