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
  namespace ScalarConservation
  {
    /**
     * The convex limiter.
     *
     * @ingroup ScalarConservationEquations
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
       * @copydoc HyperbolicSystem::View::state_type
       */
      using state_type = typename HyperbolicSystemView::state_type;

      /**
       * @copydoc HyperbolicSystem::View::n_precomputed_values
       */
      static constexpr unsigned int n_precomputed_values =
          HyperbolicSystemView::n_precomputed_values;

      /**
       * @copydoc HyperbolicSystem::View::flux_contribution_type
       */
      using flux_contribution_type =
          typename HyperbolicSystemView::flux_contribution_type;

      /**
       * @copydoc HyperbolicSystem::View::ScalarNumber
       */
      using ScalarNumber = typename get_value_type<Number>::type;

      /**
       * @name Stencil-based computation of bounds
       *
       * Intended usage:
       * ```
       * Limiter<dim, Number> limiter;
       * for (unsigned int i = n_internal; i < n_owned; ++i) {
       *   // ...
       *   limiter.reset(i, U_i);
       *   for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
       *     // ...
       *     limiter.accumulate(js, U_i, U_j, pre_i, pre_j, scaled_c_ij,
       * beta_ij);
       *   }
       *   limiter.apply_relaxation(hd_i, limiter_relaxation_factor_);
       *   limiter.bounds();
       * }
       * ```
       */
      //@{

      /**
       * The number of stored entries in the bounds array.
       */
      static constexpr unsigned int n_bounds = 2;

      /**
       * Array type used to store accumulated bounds.
       */
      using Bounds = std::array<Number, n_bounds>;

      /**
       * Constructor taking a HyperbolicSystem instance as argument
       */
      Limiter(const HyperbolicSystem &hyperbolic_system,
              const MultiComponentVector<ScalarNumber, n_precomputed_values>
                  &precomputed_values)
          : hyperbolic_system(hyperbolic_system)
          , precomputed_values(precomputed_values)
      {
      }

      /**
       * Reset temporary storage
       */
      void reset(const unsigned int /*i*/, const state_type & /*U_i*/);

      /**
       * When looping over the sparsity row, add the contribution associated
       * with the neighboring state U_j.
       */
      void accumulate(const unsigned int *js,
                      const state_type &U_i,
                      const state_type &U_j,
                      const flux_contribution_type &flux_i,
                      const flux_contribution_type &flux_j,
                      const dealii::Tensor<1, dim, Number> &scaled_c_ij,
                      const Number beta_ij);

      /**
       * Apply relaxation.
       */
      void apply_relaxation(const Number hd_i,
                            const ScalarNumber factor = ScalarNumber(2.));

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
       */
      static std::tuple<Number, bool>
      limit(const HyperbolicSystemView &hyperbolic_system,
            const Bounds &bounds,
            const state_type &U,
            const state_type &P,
            const ScalarNumber newton_tolerance,
            const unsigned int newton_max_iter,
            const Number t_min = Number(0.),
            const Number t_max = Number(1.));

      //*}
      /**
       * @name Verify invariant domain property
       */
      //@{

      /**
       * Returns whether the state @p U is located in the invariant domain
       * described by @p bounds. If @p U is a vectorized state then the
       * function returns true if all vectorized values are located in the
       * invariant domain.
       */
      static bool
      is_in_invariant_domain(const HyperbolicSystemView & /*hyperbolic_system*/,
                             const Bounds & /*bounds*/,
                             const state_type & /*U*/);

    private:
      //*}
      /** @name */
      //@{
      const HyperbolicSystemView hyperbolic_system;

      const MultiComponentVector<ScalarNumber, n_precomputed_values>
          &precomputed_values;

      Bounds bounds_;

      Number u_relaxation_numerator;
      Number u_relaxation_denominator;
      //@}
    };


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void
    Limiter<dim, Number>::reset(const unsigned int /*i*/,
                                const state_type & /*U_i*/)
    {
      /* Bounds: */

      auto &[u_min, u_max] = bounds_;

      u_min = Number(std::numeric_limits<ScalarNumber>::max());
      u_max = Number(0.);

      /* Relaxation: */

      u_relaxation_numerator = Number(0.);
      u_relaxation_denominator = Number(0.);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void Limiter<dim, Number>::accumulate(
        const unsigned int * /*js*/,
        const state_type &U_i,
        const state_type &U_j,
        const flux_contribution_type &flux_i,
        const flux_contribution_type &flux_j,
        const dealii::Tensor<1, dim, Number> &scaled_c_ij,
        const Number beta_ij)
    {
      /* Bounds: */

      auto &[u_min, u_max] = bounds_;

      const auto u_i = hyperbolic_system.state(U_i);
      const auto u_j = hyperbolic_system.state(U_j);

      const auto U_ij_bar =
          ScalarNumber(0.5) * (U_i + U_j) -
          ScalarNumber(0.5) * contract(add(flux_j, -flux_i), scaled_c_ij);

      const auto u_ij_bar = hyperbolic_system.state(U_ij_bar);

      /* Bounds: */

      u_min = std::min(u_min, u_ij_bar);
      u_max = std::max(u_max, u_ij_bar);

      /* Relaxation: */

      u_relaxation_numerator += beta_ij * (u_i + u_j);
      u_relaxation_denominator += std::abs(beta_ij);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void
    Limiter<dim, Number>::apply_relaxation(Number hd_i, ScalarNumber factor)
    {
      auto &[u_min, u_max] = bounds_;

      /* Use r_i = factor * (m_i / |Omega|) ^ (1.5 / d): */

      Number r_i = std::sqrt(hd_i);                              // in 3D: ^ 3/6
      if constexpr (dim == 2)                                    //
        r_i = dealii::Utilities::fixed_power<3>(std::sqrt(r_i)); // in 2D: ^ 3/4
      else if constexpr (dim == 1)                               //
        r_i = dealii::Utilities::fixed_power<3>(r_i);            // in 1D: ^ 3/2
      r_i *= factor;

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
      const Number u_relaxation =
          std::abs(u_relaxation_numerator) /
          (std::abs(u_relaxation_denominator) + Number(eps));

      u_min = std::max((Number(1.) - r_i) * u_min,
                       u_min - ScalarNumber(2.) * u_relaxation);

      u_max = std::min((Number(1.) + r_i) * u_max,
                       u_max + ScalarNumber(2.) * u_relaxation);
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


  } // namespace ScalarConservation
} // namespace ryujin
