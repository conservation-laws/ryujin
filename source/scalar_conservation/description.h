//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
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
  namespace ScalarConservation
  {
    /**
     * This struct contains all equation specific classes describing the
     * chosen hyperbolic system, the indicator, the limiter and
     * (approximate) Riemann solver.
     *
     * A scalar conservation equation for a scalar unknown u with a
     * user-specified flux depending on the state u.
     *
     * The parabolic subsystem is chosen to be the identity.
     *
     * @note We group all of these templates together in this struct so
     * that we only need to add a single template parameter to the all the
     * algorithm classes, such as HyperbolicModule.
     *
     * @ingroup ScalarConservationEquations
     */
    struct Description {
      using HyperbolicSystem = ScalarConservation::HyperbolicSystem;

      template <int dim, typename Number = double>
      using HyperbolicSystemView =
          ScalarConservation::HyperbolicSystemView<dim, Number>;

      using ParabolicSystem = ScalarConservation::ParabolicSystem;

      template <int dim, typename Number = double>
      using ParabolicSolver = ryujin::StubSolver<Description, dim, Number>;

      template <int dim, typename Number = double>
      using Indicator = ScalarConservation::Indicator<dim, Number>;

      template <int dim, typename Number = double>
      using Limiter = ScalarConservation::Limiter<dim, Number>;

      template <int dim, typename Number = double>
      using RiemannSolver = ScalarConservation::RiemannSolver<dim, Number>;
    };
  } // namespace ScalarConservation
} // namespace ryujin
