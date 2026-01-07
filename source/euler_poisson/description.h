//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 by the ryujin authors
//

#pragma once

#include "../euler/hyperbolic_system.h"
#include "../euler/indicator.h"
#include "../euler/limiter.h"
#include "../euler/riemann_solver.h"
#include "parabolic_module.h"
#include "parabolic_system.h"

namespace ryujin
{
  namespace EulerPoisson
  {
    struct Description {
      using HyperbolicSystem = Euler::HyperbolicSystem;

      template <int dim, typename Number = double>
      using HyperbolicSystemView = Euler::HyperbolicSystemView<dim, Number>;

      using ParabolicSystem = EulerPoisson::ParabolicSystem;

      template <int dim, typename Number = double>
      using ParabolicModule =
          EulerPoisson::ParabolicModule<Description, dim, Number>;

      template <int dim, typename Number = double>
      using Indicator = Euler::Indicator<dim, Number>;

      template <int dim, typename Number = double>
      using Limiter = Euler::Limiter<dim, Number>;

      template <int dim, typename Number = double>
      using RiemannSolver = Euler::RiemannSolver<dim, Number>;
    };
  } // namespace EulerPoisson
} // namespace ryujin
