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
  namespace ShallowWater
  {
    /**
     * @todo documentation
     *
     * @ingroup ShallowWaterEquations
     */
    template <int dim, typename Number = double>
    class Limiter
    {
    public:
      /**
       * @copydoc HyperbolicSystem::problem_dimension
       */
      static constexpr unsigned int problem_dimension =
          HyperbolicSystem::problem_dimension<dim>;

      /**
       * @copydoc HyperbolicSystem::state_type
       */
      using state_type = HyperbolicSystem::state_type<dim, Number>;

      /**
       * @copydoc HyperbolicSystem::n_precomputed_values
       */
      static constexpr unsigned int n_precomputed_values =
          HyperbolicSystem::n_precomputed_values<dim>;

      /**
       * @copydoc HyperbolicSystem::precomputed_type
       */
      using precomputed_type = HyperbolicSystem::precomputed_type<dim, Number>;

      /**
       * @copydoc HyperbolicSystem::flux_contribution_type
       */
      using flux_contribution_type =
          HyperbolicSystem::flux_contribution_type<dim, Number>;

      /**
       * @copydoc HyperbolicSystem::ScalarNumber
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
      static constexpr unsigned int n_bounds = 3;

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
      void reset(const unsigned int i);

      /**
       * When looping over the sparsity row, add the contribution associated
       * with the neighboring state U_j.
       */
      void accumulate(const unsigned int *js,
                      const state_type &U_i,
                      const state_type &U_j,
                      const flux_contribution_type &prec_i,
                      const flux_contribution_type &prec_j,
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
      limit(const HyperbolicSystem &hyperbolic_system,
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
       * Returns whether the state @ref U is located in the invariant domain
       * described by @ref bounds. If @ref U is a vectorized state then the
       * function returns true if all vectorized values are located in the
       * invariant domain.
       */
      static bool
      is_in_invariant_domain(const HyperbolicSystem &hyperbolic_system,
                             const Bounds &bounds,
                             const state_type &U);

    private:
      //*}
      /** @name */
      //@{

      const HyperbolicSystem &hyperbolic_system;

      const MultiComponentVector<ScalarNumber, n_precomputed_values>
          &precomputed_values;

      Bounds bounds_;

      /* for relaxation */

      Number h_relaxation_numerator;
      Number kin_relaxation_numerator;
      Number relaxation_denominator;

      //@}
    };


    /* Inline definitions */


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void
    Limiter<dim, Number>::reset(unsigned int /*i*/)
    {
      auto &[h_min, h_max, kin_max] = bounds_;

      h_min = Number(std::numeric_limits<ScalarNumber>::max());
      h_max = Number(0.);
      kin_max = Number(0.);

      h_relaxation_numerator = Number(0.);
      kin_relaxation_numerator = Number(0.);
      relaxation_denominator = Number(0.);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void Limiter<dim, Number>::accumulate(
        const unsigned int * /*js*/,
        const state_type & /*U_i*/,
        const state_type & /*U_j*/,
        const flux_contribution_type &prec_i,
        const flux_contribution_type &prec_j,
        const dealii::Tensor<1, dim, Number> &scaled_c_ij,
        const Number beta_ij)
    {
      /* The bar states: */

      const auto &[U_i, Z_i] = prec_i;
      const auto &[U_j, Z_j] = prec_j;
      const auto U_star_ij = hyperbolic_system.star_state(U_i, Z_i, Z_j);
      const auto U_star_ji = hyperbolic_system.star_state(U_j, Z_j, Z_i);
      const auto f_star_ij = hyperbolic_system.f(U_star_ij);
      const auto f_star_ji = hyperbolic_system.f(U_star_ji);

      const auto U_ij_bar = ScalarNumber(0.5) *
                            (U_star_ij + U_star_ji +
                             contract(add(f_star_ij, -f_star_ji), scaled_c_ij));

      /* Bounds: */

      auto &[h_min, h_max, kin_max] = bounds_;

      const auto h_bar_ij = hyperbolic_system.water_depth(U_ij_bar);
      h_min = std::min(h_min, h_bar_ij);
      h_max = std::max(h_max, h_bar_ij);

      const auto kin_bar_ij = hyperbolic_system.kinetic_energy(U_ij_bar);
      kin_max = std::max(kin_max, kin_bar_ij);

      /* Relaxation: */

      const auto h_i = hyperbolic_system.water_depth(U_i);
      const auto h_j = hyperbolic_system.water_depth(U_j);
      h_relaxation_numerator += beta_ij * (h_i + h_j);

      const auto kin_i = hyperbolic_system.kinetic_energy(U_i);
      const auto kin_j = hyperbolic_system.kinetic_energy(U_j);
      kin_relaxation_numerator += beta_ij * (kin_i + kin_j);
      relaxation_denominator += beta_ij;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void
    Limiter<dim, Number>::apply_relaxation(Number hd_i, ScalarNumber factor)
    {
      auto &[h_min, h_max, kin_max] = bounds_;

      /* Use r_i = factor * (m_i / |Omega|) ^ (1.5 / d): */

      Number r_i = std::sqrt(hd_i);                              // in 3D: ^ 3/6
      if constexpr (dim == 2)                                    //
        r_i = dealii::Utilities::fixed_power<3>(std::sqrt(r_i)); // in 2D: ^ 3/4
      else if constexpr (dim == 1)                               //
        r_i = dealii::Utilities::fixed_power<3>(r_i);            // in 1D: ^ 3/2
      r_i *= factor;

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();

      const Number h_relaxation =
          std::abs(h_relaxation_numerator) /
          (std::abs(relaxation_denominator) + Number(eps));

      h_min = std::max((Number(1.) - r_i) * h_min, h_min - h_relaxation);
      h_max = std::min((Number(1.) + r_i) * h_max, h_max + h_relaxation);

      const Number kin_relaxation =
          std::abs(kin_relaxation_numerator) /
          (std::abs(relaxation_denominator) + Number(eps));

      kin_max =
          std::min((Number(1.) + r_i) * kin_max, kin_max + kin_relaxation);
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
        const HyperbolicSystem & /*hyperbolic_system*/,
        const Bounds & /*bounds*/,
        const state_type & /*U*/)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
      __builtin_trap();
      return true;
    }

  } // namespace ShallowWater
} // namespace ryujin
