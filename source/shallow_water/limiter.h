//
// SPDX-License-Identifier: Apache-2.0
// [LANL Copyright Statement]
// Copyright (C) 2023 - 2025 by the ryujin authors
// Copyright (C) 2023 - 2024 by Triad National Security, LLC
//

#pragma once

#include <compile_time_options.h>

#include "hyperbolic_system.h"

#include <multicomponent_vector.h>
#include <newton.h>
#include <observer_pointer.h>

namespace ryujin
{
  namespace ShallowWater
  {
    template <int dim, typename Number = double>
    class LimiterView;

    /**
     * The convex limiter.
     *
     * @ingroup ShallowWaterEquations
     */
    template <typename ScalarNumber = double>
    class Limiter : public dealii::ParameterAcceptor
    {
    public:
      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      /**
       * Alias for the view on the limiter for a given dimension @p dim
       * and choice of number type @p Number.
       */
      template <int dim, typename Number = double>
      using View = LimiterView<dim, Number>;

      //@}
      /**
       * @name Constructor and setup
       */
      //@{

      /**
       * Constructor.
       */
      Limiter(const HyperbolicSystem &hyperbolic_system,
              const std::string &subsection = "/Limiter")
          : ParameterAcceptor(subsection)
          , hyperbolic_system_(&hyperbolic_system)
      {
        iterations_ = 2;
        add_parameter(
            "iterations", iterations_, "Number of limiter iterations");

        if constexpr (std::is_same_v<ScalarNumber, double>)
          newton_tolerance_ = 1.e-10;
        else
          newton_tolerance_ = 1.e-4;
        add_parameter("newton tolerance",
                      newton_tolerance_,
                      "Tolerance for the quadratic newton stopping criterion");

        newton_max_iterations_ = 2;
        add_parameter("newton max iterations",
                      newton_max_iterations_,
                      "Maximal number of quadratic newton iterations performed "
                      "during limiting");

        relaxation_factor_ = ScalarNumber(1.);
        add_parameter("relaxation factor",
                      relaxation_factor_,
                      "Factor for scaling the relaxation window with r_i = "
                      "factor * (m_i/|Omega|)^(1.5/d).");
      }

      //@}
      /**
       * @name Information and statistics
       */
      //@{

      ACCESSOR_READ_ONLY(iterations);
      ACCESSOR_READ_ONLY(newton_tolerance);
      ACCESSOR_READ_ONLY(newton_max_iterations);
      ACCESSOR_READ_ONLY(relaxation_factor);

      /**
       * Return a view on the Limiter for a given dimension @p dim and
       * choice of number type @p Number (which can be a scalar float, or
       * double, as well as a VectorizedArray holding packed scalars).
       */
      template <int dim, typename Number>
      auto view() const
      {
        return View<dim, Number>{
            hyperbolic_system_->template view<dim, Number>(), *this};
      }

    private:
      //@}
      /**
       * @name Run time options
       */
      //@{

      unsigned int iterations_;
      ScalarNumber newton_tolerance_;
      unsigned int newton_max_iterations_;
      ScalarNumber relaxation_factor_;

      //@}
      /**
       * @name Internal data
       */
      //@{

      dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;

      //@}
    };


    /**
     * A view of the Limiter that makes the interface available for a given
     * dimension @p dim and choice of number type @p Number (which can be a
     * scalar float, or double, as well as a VectorizedArray holding packed
     * scalars).
     *
     * @ingroup ShallowWaterEquations
     */
    template <int dim, typename Number>
    class LimiterView
    {
    public:
      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      using View = HyperbolicSystemView<dim, Number>;

      using ScalarNumber = typename View::ScalarNumber;

      static constexpr auto problem_dimension = View::problem_dimension;

      using state_type = typename View::state_type;

      using flux_contribution_type = typename View::flux_contribution_type;

      using precomputed_type = typename View::precomputed_type;

      using PrecomputedVectorView = typename View::PrecomputedVectorView;

      //@}
      /**
       * @name Computation and manipulation of bounds
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
       * Constructor taking a HyperbolicSystemView and a Limiter
       * object as arguments
       */
      LimiterView(const View &view, const Limiter<ScalarNumber> &limiter)
          : view_(view)
          , limiter_(limiter)
      {
      }

      /**
       * Given a state @p U_i and an index @p i return "strict" bounds,
       * i.e., a minimal convex set containing the state.
       */
      Bounds projection_bounds_from_state(const PrecomputedVectorView &pv,
                                          const unsigned int i,
                                          const state_type &U_i) const;

      /**
       * Given two bounds bounds_left, bounds_right, this function computes
       * a larger, combined set of bounds that this is a (convex) superset
       * of the two.
       */
      Bounds combine_bounds(const Bounds &bounds_left,
                            const Bounds &bounds_right) const;

      /**
       * This function applies a relaxation to a given a (strict) bound @p
       * bounds using a non dimensionalized measure @p hd (that should
       * scale as $h^d$, where $h$ is the local mesh size). This is done
       * for the case of the shallow water equations by multiplying maximum
       * bounds with $(1+r)$ and minimum bounds with $(1-r)$.
       */
      Bounds fully_relax_bounds(const Bounds &bounds, const Number &hd) const;

      //@}
      /**
       * @name Stencil-based computation of bounds
       *
       * Intended usage:
       * ```
       * LimiterView<dim, Number> limiter_view;
       * for (unsigned int i = n_internal; i < n_owned; ++i) {
       *   // ...
       *   limiter_view.reset(pv, i, U_i, flux_i);
       *   for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
       *     // ...
       *     limiter_view.accumulate(pv, js, U_j, flux_j, scaled_c_ij,
       * affine_shift);
       *   }
       *   limiter_view.bounds(hd_i);
       * }
       * ```
       */
      //@{

      /**
       * Reset temporary storage
       */
      void reset(const PrecomputedVectorView &pv,
                 const unsigned int i,
                 const state_type &U_i,
                 const flux_contribution_type &flux_i);

      /**
       * When looping over the sparsity row, add the contribution associated
       * with the neighboring state U_j.
       */
      void accumulate(const PrecomputedVectorView &pv,
                      const state_type &U_j,
                      const state_type &U_star_ij,
                      const state_type &U_star_ji,
                      const dealii::Tensor<1, dim, Number> &scaled_c_ij,
                      const state_type &affine_shift);

      /**
       * Return the computed bounds (with relaxation applied).
       */
      Bounds bounds(const Number hd_i) const;

      //@}
      /**
       * @name Convex limiter
       */
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
                                     const Number t_max = Number(1.)) const;

    private:
      //@}
      /**
       * @name Internal data
       */
      //@{

      const View view_;
      const Limiter<ScalarNumber> &limiter_;

      state_type U_i_;

      Bounds bounds_;

      /* for relaxation */

      Number h_relaxation_numerator_;
      Number v2_relaxation_numerator_;
      Number relaxation_denominator_;

      //@}
    };


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    LimiterView<dim, Number>::projection_bounds_from_state(
        const PrecomputedVectorView & /*pv*/,
        const unsigned int /*i*/,
        const state_type &U_i) const -> Bounds
    {
      const auto h_i = view_.water_depth(U_i);
      const auto v_i =
          view_.momentum(U_i) * view_.inverse_water_depth_mollified(U_i);
      const auto v2_i = v_i.norm_square();

      return {/*h_min*/ h_i, /*h_max*/ h_i, /*v2_max*/ v2_i};
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto LimiterView<dim, Number>::combine_bounds(
        const Bounds &bounds_l, const Bounds &bounds_r) const -> Bounds
    {
      const auto &[h_min_l, h_max_l, v2_max_l] = bounds_l;
      const auto &[h_min_r, h_max_r, v2_max_r] = bounds_r;

      return {std::min(h_min_l, h_min_r),
              std::max(h_max_l, h_max_r),
              std::max(v2_max_l, v2_max_r)};
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    LimiterView<dim, Number>::fully_relax_bounds(const Bounds &bounds,
                                                 const Number &hd) const
        -> Bounds
    {
      auto relaxed_bounds = bounds;
      auto &[h_min, h_max, v2_max] = relaxed_bounds;

      /* Use r = factor * (m_i / |Omega|) ^ (1.5 / d): */

      Number r = std::sqrt(hd);                              // in 3D: ^ 3/6
      if constexpr (dim == 2)                                //
        r = dealii::Utilities::fixed_power<3>(std::sqrt(r)); // in 2D: ^ 3/4
      else if constexpr (dim == 1)                           //
        r = dealii::Utilities::fixed_power<3>(r);            // in 1D: ^ 3/2
      r *= limiter_.relaxation_factor();

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
      h_min *= std::max((Number(1.) - r), Number(eps));
      h_max *= (Number(1.) + r);
      v2_max *= (Number(1.) + r);

      return relaxed_bounds;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void
    LimiterView<dim, Number>::reset(const PrecomputedVectorView & /*pv*/,
                                    unsigned int /*i*/,
                                    const state_type &U_i,
                                    const flux_contribution_type & /*flux_i*/)
    {
      U_i_ = U_i;

      auto &[h_min, h_max, v2_max] = bounds_;

      h_min = Number(std::numeric_limits<ScalarNumber>::max());
      h_max = Number(0.);
      v2_max = Number(0.);

      h_relaxation_numerator_ = Number(0.);
      v2_relaxation_numerator_ = Number(0.);
      relaxation_denominator_ = Number(0.);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void LimiterView<dim, Number>::accumulate(
        const PrecomputedVectorView & /*pv*/,
        const state_type &U_j,
        const state_type &U_star_ij,
        const state_type &U_star_ji,
        const dealii::Tensor<1, dim, Number> &scaled_c_ij,
        const state_type &affine_shift)
    {
      /* The bar states: */

      const auto f_star_ij = view_.f(U_star_ij);
      const auto f_star_ji = view_.f(U_star_ji);

      /* bar state shifted by an affine shift: */
      const auto U_ij_bar =
          ScalarNumber(0.5) *
              (U_star_ij + U_star_ji +
               contract(add(f_star_ij, -f_star_ji), scaled_c_ij)) +
          affine_shift;

      /* Bounds: */

      auto &[h_min, h_max, v2_max] = bounds_;

      const auto h_bar_ij = view_.water_depth(U_ij_bar);
      h_min = std::min(h_min, h_bar_ij);
      h_max = std::max(h_max, h_bar_ij);

      const auto v_bar_ij = view_.momentum(U_ij_bar) *
                            view_.inverse_water_depth_mollified(U_ij_bar);
      const auto v2_bar_ij = v_bar_ij.norm_square();
      v2_max = std::max(v2_max, v2_bar_ij);

      /* Relaxation: */

      /* Use a uniform weight. */
      const auto beta_ij = Number(1.);

      relaxation_denominator_ += std::abs(beta_ij);

      const auto h_i = view_.water_depth(U_i_);
      const auto h_j = view_.water_depth(U_j);
      h_relaxation_numerator_ += beta_ij * (h_i + h_j);

      const auto vel_i =
          view_.momentum(U_i_) * view_.inverse_water_depth_mollified(U_i_);
      const auto vel_j =
          view_.momentum(U_j) * view_.inverse_water_depth_mollified(U_j);
      v2_relaxation_numerator_ +=
          beta_ij * (-vel_i.norm_square() + vel_j.norm_square());
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    LimiterView<dim, Number>::bounds(const Number hd_i) const -> Bounds
    {
      const auto &[h_min, h_max, v2_max] = bounds_;

      auto relaxed_bounds = fully_relax_bounds(bounds_, hd_i);
      auto &[h_min_relaxed, h_max_relaxed, v2_max_relaxed] = relaxed_bounds;

      /* Apply a stricter window: */

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();

      const Number h_relaxed = ScalarNumber(2. * limiter_.relaxation_factor()) *
                               std::abs(h_relaxation_numerator_) /
                               (relaxation_denominator_ + Number(eps));

      const Number v2_relaxed =
          ScalarNumber(2. * limiter_.relaxation_factor()) *
          std::abs(v2_relaxation_numerator_) /
          (relaxation_denominator_ + Number(eps));

      h_min_relaxed = std::max(h_min_relaxed, h_min - h_relaxed);
      h_max_relaxed = std::min(h_max_relaxed, h_max + h_relaxed);
      v2_max_relaxed = std::min(v2_max_relaxed, v2_max + v2_relaxed);

      return relaxed_bounds;
    }
  } // namespace ShallowWater
} // namespace ryujin
