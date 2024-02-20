//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "riemann_solver.h"

#include <simd.h>

#include <random>

// #define DEBUG_RIEMANN_SOLVER

namespace ryujin
{
  namespace ScalarConservation
  {
    template <int dim, typename Number>
    Number RiemannSolver<dim, Number>::compute(
        const Number &u_i,
        const Number &u_j,
        const precomputed_state_type &prec_i,
        const precomputed_state_type &prec_j,
        const dealii::Tensor<1, dim, Number> &n_ij) const
    {
      const auto &view = hyperbolic_system.view<dim, Number>();

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

      if (parameters.use_greedy_wavespeed()) {
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

      /*
       * Thread-local helper lambda to generate a random number in [0,1]:
       */

      thread_local static const auto draw = []() {
        static std::random_device random_device;
        static auto generator = std::default_random_engine(random_device());
        static std::uniform_real_distribution<ScalarNumber> dist(0., 1.);

        if constexpr (std::is_same_v<ScalarNumber, Number>) {
          /*
           * Scalar quantity:
           */
          return dist(generator);

        } else {
          /*
           * Populate a vectorized array:
           */
          Number result;
          for (unsigned int s = 0; s < Number::size(); ++s)
            result[s] = dist(generator);
          return result;
        }
      };

      /*
       * Helper functions for enforcing entropy inequalities:
       */

      const auto enforce_entropy = [&](const Number &k) {
        const Number f_k = view.flux_function(k) * n_ij;

#ifdef DEBUG_RIEMANN_SOLVER
        std::cout << "k    = " << k << std::endl;
        std::cout << "f_k  = " << f_k << std::endl;
#endif

        const Number eta_i = view.kruzkov_entropy(k, u_i);
        const Number q_i =
            view.kruzkov_entropy_derivative(k, u_i) * (f_i - f_k);

        const Number eta_j = view.kruzkov_entropy(k, u_j);
        const Number q_j =
            view.kruzkov_entropy_derivative(k, u_j) * (f_j - f_k);

        const Number a = u_i + u_j - ScalarNumber(2.) * k;
        const Number b = f_j - f_i;
        const Number c = eta_i + eta_j;
        const Number d = q_j - q_i;

        /*
         * FIXME: Ordinarily, lambda_left and lambda_right would be
         * computed without taking the absolute value of the numerator.
         * (The denominator is - in the absence of rounding errors - always
         * nonnegative. The numerator has a sign.)
         * But empirically it turns out that taking the absolute value and
         * letting both estimates participate in the maximal wavespeed
         * estimate helps a lot.
         */
        const Number lambda_left = std::abs(d + b) / (std::abs(c + a) + h2);
        const Number lambda_right = std::abs(d - b) / (std::abs(c - a) + h2);

#ifdef DEBUG_RIEMANN_SOLVER
        std::cout << "   left  wavespeed   = " << lambda_left << std::endl;
        std::cout << "   right wavespeed   = " << lambda_right << std::endl;
#endif
        lambda_max = std::max(lambda_max, lambda_left);
        lambda_max = std::max(lambda_max, lambda_right);
      };


      if (parameters.use_averaged_entropy()) {
        const Number k = ScalarNumber(0.5) * (u_i + u_j);
        enforce_entropy(k);
      }

      const unsigned int n_entropies = parameters.random_entropies();
      for (unsigned int i = 0; i < n_entropies; ++i) {
        const Number factor = draw();
        const Number k = factor * u_i + (Number(1.) - factor) * u_j;
        enforce_entropy(k);
      }

#ifdef DEBUG_RIEMANN_SOLVER
      std::cout << "-> lambda_max        = " << lambda_max << std::endl;
#endif
      return lambda_max;
    }

  } // namespace ScalarConservation
} // namespace ryujin
