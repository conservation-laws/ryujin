//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "riemann_solver.h"

#include <simd.h>

// #define DEBUG_RIEMANN_SOLVER

namespace ryujin
{
  namespace ScalarConservation
  {
    template <int dim, typename Number>
    Number RiemannSolver<dim, Number>::compute(
        const state_type &U_i,
        const state_type &U_j,
        const unsigned int i,
        const unsigned int *js,
        const dealii::Tensor<1, dim, Number> &n_ij) const
    {
      using pst =
          typename HyperbolicSystem::View<dim, Number>::precomputed_state_type;

      const auto view = hyperbolic_system.template view<dim, Number>();

      const auto u_i = hyperbolic_system.state(U_i);
      const auto u_j = hyperbolic_system.state(U_j);

      const auto &pv = precomputed_values;
      const auto prec_i = pv.template get_tensor<Number, pst>(i);
      const auto prec_j = pv.template get_tensor<Number, pst>(js);

      /* Project all fluxes to 1D: */
      const Number f_i = view.construct_flux_tensor(prec_i) * n_ij;
      const Number f_j = view.construct_flux_tensor(prec_j) * n_ij;
      const Number df_i = view.construct_flux_gradient_tensor(prec_i) * n_ij;
      const Number df_j = view.construct_flux_gradient_tensor(prec_j) * n_ij;

      const auto h2 = Number(2. * view.derivative_approximation_delta());

#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << "\nu_i  = " << u_i << std::endl;
      std::cout << "u_j  = " << u_j << std::endl;
      std::cout << "f_i  = " << f_i << std::endl;
      std::cout << "f_j  = " << f_j << std::endl;
      std::cout << "df_i = " << df_i << std::endl;
      std::cout << "df_j = " << df_j << std::endl;
#endif

      /*
       * The Roe average with a regularization based on $h$ which is the
       * step size used for the central difference approximation of f'(u).
       *
       * The regularization max(|u_i - u_j|, 2 * h) ensures that the
       * quotient approximates the derivative f'( (u_i + u_j)/2 ) to the
       * same precision that we use to compute f'(u_i) and f'(u_j) in the
       * FunctionParser (via a central difference approximation).
       *
       * This implies that in contrast to the actual limit of the
       * difference quotient we will approach 0 as |u_j - u_i| goes to
       * zero. We fix this by taking the maximum with our approximation of
       * f'(u_i) and f'(u_j) further down below.
       */

      auto lambda_max = std::abs(f_i - f_j) / std::max(std::abs(u_i - u_j), h2);
#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << "   Roe average       = " << lambda_max << std::endl;
#endif

      constexpr auto gte = dealii::SIMDComparison::greater_than_or_equal;

      if (view.riemann_solver_greedy_wavespeed()) {
        /*
         * In case of a greedy estimate we make sure that we always use the
         * Roe average and only fall back to the derivative approximation
         * when u_i and u_j are close to each other within 2h:
         */
        lambda_max = dealii::compare_and_apply_mask<gte>(
            std::abs(u_i - u_j),
            h2,
            lambda_max,
            /* Approximate derivative in centerpoint: */
            std::abs(ScalarNumber(0.5) * (df_i + df_j)));
#ifdef DEBUG_RIEMANN_SOLVER
        std::cout << "   interpolated      = "
                  << std::abs(ScalarNumber(0.5) * (df_i + df_j)) << std::endl;
#endif

      } else {
        /*
         * Always take the maximum with |f'(u_i)| and |f'(u_j)|.
         *
         * For convex fluxes this implies that lambda_max is indeed the
         * maximal wavespeed of the system. See Example 79.17 in reference
         * @cite ErnGuermond2021.
         */
        lambda_max = std::max(lambda_max, std::abs(df_i));
        lambda_max = std::max(lambda_max, std::abs(df_j));
#ifdef DEBUG_RIEMANN_SOLVER
        std::cout << "   left  derivative  = " << std::abs(df_i) << std::endl;
        std::cout << "   right derivative  = " << std::abs(df_j) << std::endl;
#endif
      }

      if (view.riemann_solver_greedy_wavespeed() ||
          !view.riemann_solver_convex_flux()) {
        /*
         * If we are approximating with a greedy wavespeed or if we have a
         * general, not necessarily convex flux we compute
         */

        Number k = ScalarNumber(0.5) * (u_i + u_j);

        if (view.riemann_solver_random_entropy()) {
          // FIXME
          __builtin_trap();
        }

        const Number f_k = view.flux_function(k) * n_ij;

#ifdef DEBUG_RIEMANN_SOLVER
        std::cout << "k    = " << k << std::endl;
        std::cout << "f_k  = " << f_k << std::endl;
#endif

        const Number eta_i = std::abs(u_i - k);
        /* We should implement and use a "copysign" intrinsic for deal.II: */
        const Number q_i = /* sign(u_i - k) * (f_i - f_k) */
            dealii::compare_and_apply_mask<gte>(u_i, k, f_i - f_k, f_k - f_i);

        const Number eta_j = std::abs(u_j - k);
        /* We should implement and use a "copysign" intrinsic for deal.II: */
        const Number q_j = /* sign(u_j - k) * (f_j - f_k) */
            dealii::compare_and_apply_mask<gte>(u_j, k, f_j - f_k, f_k - f_j);

        const Number a = u_i + u_j - ScalarNumber(2.) * k;
        const Number b = f_j - f_i;
        const Number c = eta_i + eta_j;
        const Number d = q_j - q_i;

        lambda_max = std::max(lambda_max, (d + b) / (std::abs(c + a) + h2));
        lambda_max = std::max(lambda_max, (d - b) / (std::abs(c - a) + h2));
      }

#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << "-> lambda_max        = " << lambda_max << std::endl;
#endif
      return lambda_max;
    }

  } // namespace ScalarConservation
} // namespace ryujin
