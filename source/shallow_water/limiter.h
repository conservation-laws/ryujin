//
// SPDX-License-Identifier: MIT or BSD-3-Clause
// [LANL Copyright Statement]
// Copyright (C) 2020 - 2023 by the ryujin authors
// Copyright (C) 2023 - 2023 by Triad National Security, LLC
//

#pragma once

#include "hyperbolic_system.h"

#include <compile_time_options.h>
#include <multicomponent_vector.h>
#include <newton.h>

namespace ryujin
{
  namespace ShallowWater
  {
    /**
     * The convex limiter.
     *
     * @ingroup ShallowWaterEquations
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
       *   limiter.reset(i, U_i, flux_i);
       *   for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
       *     // ...
       *     limiter.accumulate(
       *       js, U_j, flux_j, scaled_c_ij, beta_ij, affine_shift);
       *   }
       *   limiter.bounds(hd_i);
       * }
       * ```
       */
      //@{

      /**
       * The number of stored entries in the bounds array.
       */
      static constexpr unsigned int n_bounds = 5;

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
              const ScalarNumber newton_tol,
              const unsigned int newton_max_iter)
          : hyperbolic_system(hyperbolic_system)
          , precomputed_values(precomputed_values)
          , relaxation_factor(relaxation_factor)
          , newton_tol(newton_tol)
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
      void accumulate(const state_type &U_j,
                      const state_type &U_star_ij,
                      const state_type &U_star_ji,
                      const dealii::Tensor<1, dim, Number> &scaled_c_ij,
                      const Number &beta_ij,
                      const state_type &affine_shift);

      /**
       * Return the computed bounds (with relaxation applied).
       */
      Bounds bounds(const Number hd_i) const;

      //*}
      /** @name Convex limiter */
      //@{

      /**
       * Given a state \f$\mathbf U\f$ and an update \f$\mathbf P\f$ this
       * function computes and returns the maximal coefficient \f$t\f$,
       * obeying \f$t_{\text{min}} < t < t_{\text{max}}\f$, such that the
       * selected local minimum principles are obeyed.
       */
      std::tuple<Number, bool> limit(const Bounds &bounds,
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
       * described by @p bounds. If @p U is a vectorized state then the
       * function returns true if all vectorized values are located in the
       * invariant domain.
       */
      static bool
      is_in_invariant_domain(const HyperbolicSystemView & /*hyperbolic_system*/,
                             const Bounds & /*bounds*/,
                             const state_type & /*U*/);

    private:
      //@}
      /** @name Arguments and internal fields */
      //@{
      const HyperbolicSystemView hyperbolic_system;

      const MultiComponentVector<ScalarNumber, n_precomputed_values>
          &precomputed_values;

      ScalarNumber relaxation_factor;
      ScalarNumber newton_tol;
      unsigned int newton_max_iter;

      state_type U_i;

      Bounds bounds_;

      /* for relaxation */

      Number h_relaxation_numerator;
      Number kin_relaxation_numerator;
      Number v2_relaxation_numerator;
      Number relaxation_denominator;

      //@}
    };


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void
    Limiter<dim, Number>::reset(unsigned int /*i*/,
                                const state_type &new_U_i,
                                const flux_contribution_type & /*new_flux_i*/)
    {
      U_i = new_U_i;

      auto &[h_min, h_max, h_small, kin_max, v2_max] = bounds_;

      h_min = Number(std::numeric_limits<ScalarNumber>::max());
      h_max = Number(0.);
      h_small = Number(0.);
      kin_max = Number(0.);
      v2_max = Number(0.);

      h_relaxation_numerator = Number(0.);
      kin_relaxation_numerator = Number(0.);
      v2_relaxation_numerator = Number(0.);
      relaxation_denominator = Number(0.);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void Limiter<dim, Number>::accumulate(
        const state_type &U_j,
        const state_type &U_star_ij,
        const state_type &U_star_ji,
        const dealii::Tensor<1, dim, Number> &scaled_c_ij,
        const Number &beta_ij,
        const state_type &affine_shift)
    {
      /* The bar states: */

      const auto f_star_ij = hyperbolic_system.f(U_star_ij);
      const auto f_star_ji = hyperbolic_system.f(U_star_ji);

      auto U_ij_bar = ScalarNumber(0.5) *
                      (U_star_ij + U_star_ji +
                       contract(add(f_star_ij, -f_star_ji), scaled_c_ij));

      U_ij_bar += affine_shift;

      /* Bounds: */

      auto &[h_min, h_max, h_small, kin_max, v2_max] = bounds_;

      const auto h_bar_ij = hyperbolic_system.water_depth(U_ij_bar);
      h_min = std::min(h_min, h_bar_ij);
      h_max = std::max(h_max, h_bar_ij);

      const auto kin_bar_ij = hyperbolic_system.kinetic_energy(U_ij_bar);
      kin_max = std::max(kin_max, kin_bar_ij);

      const auto v_bar_ij =
          hyperbolic_system.momentum(U_ij_bar) *
          hyperbolic_system.inverse_water_depth_mollified(U_ij_bar);
      const auto v2_bar_ij = v_bar_ij.norm_square();
      v2_max = std::max(v2_max, v2_bar_ij);

      /* Relaxation: */

      relaxation_denominator += std::abs(beta_ij);

      const auto h_i = hyperbolic_system.water_depth(U_i);
      const auto h_j = hyperbolic_system.water_depth(U_j);
      h_relaxation_numerator += beta_ij * (h_i + h_j);

      const auto kin_i = hyperbolic_system.kinetic_energy(U_i);
      const auto kin_j = hyperbolic_system.kinetic_energy(U_j);
      kin_relaxation_numerator += beta_ij * (kin_i + kin_j);

      const auto vel_i = hyperbolic_system.momentum(U_i) *
                         hyperbolic_system.inverse_water_depth_mollified(U_i);
      const auto vel_j = hyperbolic_system.momentum(U_j) *
                         hyperbolic_system.inverse_water_depth_mollified(U_j);
      v2_relaxation_numerator +=
          beta_ij * (-vel_i.norm_square() + vel_j.norm_square());
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    Limiter<dim, Number>::bounds(const Number hd_i) const -> Bounds
    {
      auto relaxed_bounds = bounds_;
      auto &[h_min, h_max, h_small, kin_max, v2_max] = relaxed_bounds;

      /* Use r_i = factor * (m_i / |Omega|) ^ (1.5 / d): */

      Number r_i = std::sqrt(hd_i);                              // in 3D: ^ 3/6
      if constexpr (dim == 2)                                    //
        r_i = dealii::Utilities::fixed_power<3>(std::sqrt(r_i)); // in 2D: ^ 3/4
      else if constexpr (dim == 1)                               //
        r_i = dealii::Utilities::fixed_power<3>(r_i);            // in 1D: ^ 3/2
      r_i *= relaxation_factor;

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();

      const Number h_relaxed = ScalarNumber(2.) *
                               std::abs(h_relaxation_numerator) /
                               (relaxation_denominator + Number(eps));

      h_min = std::max((Number(1.) - r_i) * h_min, h_min - h_relaxed);
      h_max = std::min((Number(1.) + r_i) * h_max, h_max + h_relaxed);

      const Number kin_relaxed = ScalarNumber(2.) *
                                 std::abs(kin_relaxation_numerator) /
                                 (relaxation_denominator + Number(eps));

      kin_max = std::min((Number(1.) + r_i) * kin_max, kin_max + kin_relaxed);

      const Number v2_relaxed = ScalarNumber(2.) *
                                std::abs(v2_relaxation_numerator) /
                                (relaxation_denominator + Number(eps));

      v2_max = std::min((Number(1.) + r_i) * v2_max, v2_max + v2_relaxed);

      /* Use r_i = 0.2 * (m_i / |Omega|) ^ (1 / d): */

      r_i = hd_i;
      if constexpr (dim == 2)
        r_i = std::sqrt(hd_i);
      r_i *= hyperbolic_system.dry_state_relaxation_factor();

      h_small = hyperbolic_system.reference_water_depth() * r_i;

      return relaxed_bounds;
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


  } // namespace ShallowWater
} // namespace ryujin
