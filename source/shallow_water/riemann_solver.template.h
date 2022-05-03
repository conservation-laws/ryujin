//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "riemann_solver.h"

#include <newton.h>
#include <simd.h>

namespace ryujin
{
  template <int dim, typename Number>
  Number RiemannSolver<dim, Number>::compute(
      const std::array<Number, 4> &riemann_data_i,
      const std::array<Number, 4> &riemann_data_j)
  {
    return Number(1.);
  }


  template <int dim, typename Number>
  Number RiemannSolver<dim, Number>::compute(
      const state_type &U_i,
      const state_type &U_j,
      const dealii::Tensor<1, dim, Number> &n_ij)
  {
    return Number(1.);
  }

} /* namespace ryujin */
