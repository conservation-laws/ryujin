//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"

#include <compile_time_options.h>
#include <multicomponent_vector.h>
#include <newton.h>
#include <simd.h>

namespace ryujin
{
  namespace EulerAEOS
  {
    /**
     * The convex limiter.
     *
     * The class implements a convex limiting technique as described in
     * @cite GuermondEtAl2018,  @cite ryujin-2021-1 and
     * @cite ryujin-2023-4.
     * Given a computed set of bounds and an update direction \f$\mathbf
     * P_{ij}\f$ one can now determine a candidate \f$\tilde l_{ij}\f$ by
     * computing
     *
     * \f{align}
     *   \tilde l_{ij} = \max_{l\,\in\,[0,1]}
     *   \,\Big\{\rho_{\text{min}}\,\le\,\rho\,(\mathbf U_i +\tilde
     * l_{ij}\mathbf P_{ij})
     *   \,\le\,\rho_{\text{max}},\quad
     *   \phi_{\text{min}}\,\le\,\phi\,(\mathbf U_{i}+\tilde l_{ij}\mathbf
     * P_{ij})\Big\}, \f}
     *
     * where \f$\psi\f$ denots the specific entropy @cite ryujin-2021-1.
     *
     * Algorithmically this is accomplished as follows: Given an initial
     * interval \f$[t_L,t_R]\f$, where \f$t_L\f$ is a good state, we first
     * make the interval smaller ensuring the bounds on the density are
     * fulfilled. If limiting on the specific entropy is selected we then
     * then perform a quadratic Newton iteration (updating \f$[t_L,t_R]\f$
     * solving for the root of a 3-convex function
     * \f{align}
     *     \Psi(\mathbf U)\;=\;\rho^{\gamma+1}(\mathbf U)\,\big(\phi(\mathbf
     * U)-\phi_{\text{min}}\big). \f}
     *
     * @ingroup EulerEquations
     */
    template <int dim, typename Number = double>
    class Limiter
    {
    public:
      /**
       * @copydoc HyperbolicSystem::View
       */
      using HyperbolicSystemView = HyperbolicSystem::View<dim, Number>;

      /**
       * @copydoc HyperbolicSystem::View::problem_dimension
       */
      static constexpr unsigned int problem_dimension =
          HyperbolicSystemView::problem_dimension;

      /**
       * @copydoc HyperbolicSystem::View::state_type
       */
      using state_type = typename HyperbolicSystemView::state_type;

      /**
       * @copydoc HyperbolicSystem::View::n_precomputed_values
       */
      static constexpr unsigned int n_precomputed_values =
          HyperbolicSystemView::n_precomputed_values;

      /**
       * @copydoc HyperbolicSystem::View::precomputed_state_type
       */
      using precomputed_state_type =
          typename HyperbolicSystemView::precomputed_state_type;

      /**
       * @copydoc HyperbolicSystem::View::flux_contribution_type
       */
      using flux_contribution_type =
          typename HyperbolicSystemView::flux_contribution_type;

      /**
       * @copydoc HyperbolicSystem::View::ScalarNumber
       */
      using ScalarNumber = typename HyperbolicSystemView::ScalarNumber;

      /**
       * @name Stencil-based computation of bounds
       *
       * Intended usage:
       * ```
       * Limiter<dim, Number> limiter;
       * for (unsigned int i = n_internal; i < n_owned; ++i) {
       *   // ...
       *   limiter.reset(i, U_i, flux_i);
       *   for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
       *     // ...
       *     limiter.accumulate(js, U_j, flux_j, scaled_c_ij, beta_ij);
       *   }
       *   limiter.apply_relaxation(hd_i);
       *   limiter.bounds();
       * }
       * ```
       */
      //@{

      /**
       * The number of stored entries in the bounds array.
       */
      static constexpr unsigned int n_bounds = 4;

      /**
       * Array type used to store accumulated bounds.
       */
      using Bounds = std::array<Number, n_bounds>;

      /**
       * Constructor taking a HyperbolicSystem instance as argument
       */
      Limiter(const HyperbolicSystem &hyperbolic_system,
              const MultiComponentVector<ScalarNumber, n_precomputed_values>
                  &precomputed_values,
            const ScalarNumber relaxation_factor,
            const ScalarNumber newton_tolerance,
            const unsigned int newton_max_iter)
          : hyperbolic_system(hyperbolic_system)
          , precomputed_values(precomputed_values)
          , relaxation_factor(relaxation_factor)
          , newton_tolerance(newton_tolerance)
          , newton_max_iter(newton_max_iter)
      {
      }

      /**
       * Reset temporary storage
       */
      void reset(const unsigned int i,
                 const state_type &U_i,
                 const flux_contribution_type &flux_i);

      /**
       * When looping over the sparsity row, add the contribution associated
       * with the neighboring state U_j.
       */
      void accumulate(const unsigned int *js,
                      const state_type &U_j,
                      const flux_contribution_type &flux_j,
                      const dealii::Tensor<1, dim, Number> &scaled_c_ij,
                      const Number beta_ij);

      /**
       * Apply relaxation.
       */
      void apply_relaxation(const Number hd_i);

      /**
       * Return the computed bounds.
       */
      const Bounds &bounds() const;

      //*}
      /** @name Convex limiter */
      //@{

      /**
       * Given a state \f$\mathbf U\f$ and an update \f$\mathbf P\f$ this
       * function computes and returns the maximal coefficient \f$t\f$,
       * obeying \f$t_{\text{min}} < t < t_{\text{max}}\f$, such that the
       * selected local minimum principles are obeyed.
       *
       * The returned boolean is set to true if the original low-order
       * update was within bounds.
       *
       * If the debug option `CHECK_BOUNDS` is set to true, then the
       * boolean is set to true if the low-order and the resulting
       * high-order update are within bounds. The latter might be violated
       * due to round-off errors when computing the limiter bounds.
       */
      std::tuple<Number, bool>
      limit(const Bounds &bounds,
            const state_type &U,
            const state_type &P,
            const Number t_min = Number(0.),
            const Number t_max = Number(1.));
      //*}
      /**
       * @name Verify invariant domain property
       */
      //@{

      /**
       * Returns whether the state @p U is located in the invariant domain
       * described by @ref bounds. If @p U is a vectorized state then the
       * function returns true if all vectorized values are located in the
       * invariant domain.
       */
      static bool
      is_in_invariant_domain(const HyperbolicSystemView &hyperbolic_system,
                             const Bounds &bounds,
                             const state_type &U);

    private:
      //@}
      /** @name Arguments and internal fields */
      //@{

      const HyperbolicSystemView hyperbolic_system;

      const MultiComponentVector<ScalarNumber, n_precomputed_values>
          &precomputed_values;

      ScalarNumber relaxation_factor;
      ScalarNumber newton_tolerance;
      unsigned int newton_max_iter;

      state_type U_i;
      flux_contribution_type flux_i;

      Bounds bounds_;

      Number rho_relaxation_numerator;
      Number rho_relaxation_denominator;
      Number s_interp_max;

      //@}
    };


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void
    Limiter<dim, Number>::reset(const unsigned int i,
                                const state_type &new_U_i,
                                const flux_contribution_type &new_flux_i)
    {
      U_i = new_U_i;
      flux_i = new_flux_i;

      /* Bounds: */

      auto &[rho_min, rho_max, s_min, gamma_min] = bounds_;

      rho_min = Number(std::numeric_limits<ScalarNumber>::max());
      rho_max = Number(0.);
      s_min = Number(std::numeric_limits<ScalarNumber>::max());

      const auto &[p_i, gamma_min_i, s_i, eta_i] =
          precomputed_values
              .template get_tensor<Number, precomputed_state_type>(i);

      gamma_min = gamma_min_i;

      /* Relaxation: */

      rho_relaxation_numerator = Number(0.);
      rho_relaxation_denominator = Number(0.);
      s_interp_max = Number(0.);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void Limiter<dim, Number>::accumulate(
        const unsigned int *js,
        const state_type &U_j,
        const flux_contribution_type &flux_j,
        const dealii::Tensor<1, dim, Number> &scaled_c_ij,
        const Number beta_ij)
    {
      /* Bounds: */

      auto &[rho_min, rho_max, s_min, gamma_min] = bounds_;

      const auto rho_i = hyperbolic_system.density(U_i);
      const auto rho_j = hyperbolic_system.density(U_j);

      const auto U_ij_bar =
          ScalarNumber(0.5) * (U_i + U_j) -
          ScalarNumber(0.5) * contract(add(flux_j, -flux_i), scaled_c_ij);

      const auto rho_ij_bar = hyperbolic_system.density(U_ij_bar);

      /* Density bounds: */

      rho_min = std::min(rho_min, rho_ij_bar);
      rho_max = std::max(rho_max, rho_ij_bar);

      /* Density relaxation: */

      rho_relaxation_numerator += beta_ij * (rho_i + rho_j);
      rho_relaxation_denominator += std::abs(beta_ij);

      /* Surrogate entropy bounds and relaxation: */

      if (hyperbolic_system.compute_strict_bounds()) {
        /*
         * Compute strict bounds precisely as outlined in @cite ryujin-2023-4
         *
         * This means, we compute
         *  - the surrogate entropy at dof j with the gamma_min of index i,
         *  - the currogate entropy of the bar state U_ij_bar
         *  - an interpolated surrogate entropy at (U_i + U_j) / 2 for
         *    bounds relaxation:
         */

        const auto s_j =
            hyperbolic_system.surrogate_specific_entropy(U_j, gamma_min);

        const auto s_ij_bar =
            hyperbolic_system.surrogate_specific_entropy(U_ij_bar, gamma_min);

        const Number s_interp = hyperbolic_system.surrogate_specific_entropy(
            (U_i + U_j) * ScalarNumber(.5), gamma_min);

        s_min = std::min(s_min, s_j);
        s_min = std::min(s_min, s_ij_bar);
        s_interp_max = std::max(s_interp_max, s_interp);

      } else {
        /*
         * Compute a cheaper bound solely relying on the diagonal s_j
         * (computed with gamma_min_j) and the surrogate entropy s_ij_bar
         * of the bar state. We use the s_ij_bar for computing the bounds
         * relaxation as well.
         */
        const auto [p_j, gamma_min_j, s_j, eta_j] =
            precomputed_values
                .template get_tensor<Number, precomputed_state_type>(js);

        const auto s_ij_bar =
            hyperbolic_system.surrogate_specific_entropy(U_ij_bar, gamma_min);

        s_min = std::min(s_min, s_j);
        s_min = std::min(s_min, s_ij_bar);
        s_interp_max = std::max(s_interp_max, s_ij_bar);
      }
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void
    Limiter<dim, Number>::apply_relaxation(Number hd_i)
    {
      auto &[rho_min, rho_max, s_min, gamma_min] = bounds_;

      /* Use r_i = factor * (m_i / |Omega|) ^ (1.5 / d): */

      Number r_i = std::sqrt(hd_i);                              // in 3D: ^ 3/6
      if constexpr (dim == 2)                                    //
        r_i = dealii::Utilities::fixed_power<3>(std::sqrt(r_i)); // in 2D: ^ 3/4
      else if constexpr (dim == 1)                               //
        r_i = dealii::Utilities::fixed_power<3>(r_i);            // in 1D: ^ 3/2
      r_i *= relaxation_factor;

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
      const Number rho_relaxation =
          std::abs(rho_relaxation_numerator) /
          (std::abs(rho_relaxation_denominator) + Number(eps));

      rho_min = std::max((Number(1.) - r_i) * rho_min,
                         rho_min - ScalarNumber(2.) * rho_relaxation);

      rho_max = std::min((Number(1.) + r_i) * rho_max,
                         rho_max + ScalarNumber(2.) * rho_relaxation);

      s_min = std::max((Number(1.) - r_i) * s_min,
                       Number(2.) * s_min - s_interp_max);

      /*
       * If we have a maximum compressibility constant, b, the maximum
       * bound for rho changes. See @cite ryujin-2023-4 for how to define
       * rho_max.
       */

      const auto numerator = (gamma_min + Number(1.)) * rho_max;
      const auto interpolation_b = hyperbolic_system.eos_interpolation_b();
      const auto denominator =
          gamma_min - Number(1.) + ScalarNumber(2.) * interpolation_b * rho_max;
      const auto upper_bound = numerator / denominator;

      rho_max = std::min(upper_bound, rho_max);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline const typename Limiter<dim, Number>::Bounds &
    Limiter<dim, Number>::bounds() const
    {
      return bounds_;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline bool
    Limiter<dim, Number>::is_in_invariant_domain(
        const HyperbolicSystemView & /*hyperbolic_system*/,
        const Bounds & /*bounds*/,
        const state_type & /*U*/)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
      __builtin_trap();
      return true;
    }

  } // namespace EulerAEOS
} // namespace ryujin
