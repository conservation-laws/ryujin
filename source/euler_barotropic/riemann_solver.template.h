//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 by the ryujin authors
//

#pragma once

#include "riemann_solver.h"

#include <newton.h>
#include <simd.h>

// #define DEBUG_RIEMANN_SOLVER

namespace ryujin
{
  namespace EulerBarotropic
  {
    template <int dim, typename Number>
    Number RiemannSolver<dim, Number>::compute(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto &[u_i, a_i] = riemann_data_i;
      const auto &[u_j, a_j] = riemann_data_j;

#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << "u_left: " << u_i << std::endl;
      std::cout << "a_left: " << a_i << std::endl;
      std::cout << "u_right: " << u_j << std::endl;
      std::cout << "a_right: " << a_j << std::endl;
#endif

      const Number lambda_max =
          std::max(std::abs(u_i) + a_i, std::abs(u_j) + a_j);
      return lambda_max;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number RiemannSolver<dim, Number>::compute(
        const state_type &U_i,
        const state_type &U_j,
        const unsigned int i,
        const unsigned int *js,
        const dealii::Tensor<1, dim, Number> &n_ij) const
    {
      const auto view = hyperbolic_system.view<dim, Number>();

      const auto &[e_i, p_i, a_i] =
          precomputed_values.template read_tensor<Number, precomputed_type>(i);

      const auto &[e_j, p_j, a_j] =
          precomputed_values.template read_tensor<Number, precomputed_type>(js);

      const auto rho_i = view.density(U_i);
      const auto rho_i_inverse = Number(1.0) / rho_i;
      const auto m_i = view.momentum(U_i);
      const auto u_i = rho_i_inverse * n_ij * m_i;

      const auto rho_j = view.density(U_j);
      const auto rho_j_inverse = Number(1.0) / rho_j;
      const auto m_j = view.momentum(U_j);
      const auto u_j = rho_j_inverse * n_ij * m_j;

      return compute(primitive_type{u_i, a_i}, primitive_type{u_j, a_j});
    }
  } // namespace EulerBarotropic
} // namespace ryujin
