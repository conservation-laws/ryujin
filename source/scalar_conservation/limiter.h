//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "hyperbolic_system.h"

#include <multicomponent_vector.h>
#include <observer_pointer.h>
#include <simd.h>

namespace ryujin
{
  namespace ScalarConservation
  {
    template <int dim, typename Number = double>
    class LimiterView;

    template <typename ScalarNumber = double>
    class Limiter : public dealii::ParameterAcceptor
    {
    public:
      Limiter(const HyperbolicSystem &hyperbolic_system,
              const std::string &subsection = "/Limiter")
          : ParameterAcceptor(subsection)
          , hyperbolic_system_(&hyperbolic_system)
      {
        iterations_ = 2;
        add_parameter(
            "iterations", iterations_, "Number of limiter iterations");

        relaxation_factor_ = ScalarNumber(1.);
        add_parameter("relaxation factor",
                      relaxation_factor_,
                      "Factor for scaling the relaxation window with r_i = "
                      "factor * (m_i/|Omega|)^(1.5/d).");
      }

      ACCESSOR_READ_ONLY(iterations);
      ACCESSOR_READ_ONLY(relaxation_factor);

      /**
       * Alias for the view on the limiter for a given dimension @p dim
       * and choice of number type @p Number.
       */
      template <int dim, typename Number = double>
      using View = LimiterView<dim, Number>;

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
      dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;
      unsigned int iterations_;
      ScalarNumber relaxation_factor_;
    };


    /**
     * The convex limiter.
     *
     * @ingroup ScalarConservationEquations
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
      static constexpr unsigned int n_bounds = 2;

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
       * scale as $h^d$, where $h$ is the local mesh size).
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
                      const unsigned int *js,
                      const state_type &U_j,
                      const flux_contribution_type &flux_j,
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
      flux_contribution_type flux_i_;

      Bounds bounds_;

      Number u_relaxation_numerator_;
      Number u_relaxation_denominator_;
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
      const auto u_i = view_.state(U_i);
      return {/*u_min*/ u_i, /*u_max*/ u_i};
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto LimiterView<dim, Number>::combine_bounds(
        const Bounds &bounds_left, const Bounds &bounds_right) const -> Bounds
    {
      const auto &[u_min_l, u_max_l] = bounds_left;
      const auto &[u_min_r, u_max_r] = bounds_right;

      return {std::min(u_min_l, u_min_r), std::max(u_max_l, u_max_r)};
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    LimiterView<dim, Number>::fully_relax_bounds(const Bounds &bounds,
                                                 const Number &hd) const
        -> Bounds
    {
      auto relaxed_bounds = bounds;
      auto &[u_min, u_max] = relaxed_bounds;

      /* Use r = factor * (m_i / |Omega|) ^ (1.5 / d): */

      Number r = std::sqrt(hd);                              // in 3D: ^ 3/6
      if constexpr (dim == 2)                                //
        r = dealii::Utilities::fixed_power<3>(std::sqrt(r)); // in 2D: ^ 3/4
      else if constexpr (dim == 1)                           //
        r = dealii::Utilities::fixed_power<3>(r);            // in 1D: ^ 3/2
      r *= limiter_.relaxation_factor();

      u_min = std::min((Number(1.) - r) * u_min, (Number(1.) + r) * u_min);
      u_max = std::max((Number(1.) + r) * u_max, (Number(1.) - r) * u_max);

      return relaxed_bounds;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void
    LimiterView<dim, Number>::reset(const PrecomputedVectorView & /*pv*/,
                                    const unsigned int /*i*/,
                                    const state_type &U_i,
                                    const flux_contribution_type &flux_i)
    {
      U_i_ = U_i;
      flux_i_ = flux_i;

      /* Bounds: */

      auto &[u_min, u_max] = bounds_;

      u_min = Number(std::numeric_limits<ScalarNumber>::max());
      u_max = Number(std::numeric_limits<ScalarNumber>::lowest());

      /* Relaxation: */

      u_relaxation_numerator_ = Number(0.);
      u_relaxation_denominator_ = Number(0.);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void LimiterView<dim, Number>::accumulate(
        const PrecomputedVectorView & /*pv*/,
        const unsigned int * /*js*/,
        const state_type &U_j,
        const flux_contribution_type &flux_j,
        const dealii::Tensor<1, dim, Number> &scaled_c_ij,
        const state_type &affine_shift)
    {
      /* Bounds: */
      auto &[u_min, u_max] = bounds_;

      const auto u_i = view_.state(U_i_);
      const auto u_j = view_.state(U_j);

      const auto U_ij_bar =
          ScalarNumber(0.5) * (U_i_ + U_j) -
          ScalarNumber(0.5) * contract(add(flux_j, -flux_i_), scaled_c_ij) +
          affine_shift;

      const auto u_ij_bar = view_.state(U_ij_bar);

      /* Bounds: */

      u_min = std::min(u_min, u_ij_bar);
      u_max = std::max(u_max, u_ij_bar);

      /* Relaxation: */

      /* Use a uniform weight. */
      const auto beta_ij = Number(1.);
      u_relaxation_numerator_ += beta_ij * (u_i + u_j);
      u_relaxation_denominator_ += std::abs(beta_ij);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    LimiterView<dim, Number>::bounds(const Number hd_i) const -> Bounds
    {
      const auto &[u_min, u_max] = bounds_;

      auto relaxed_bounds = fully_relax_bounds(bounds_, hd_i);
      auto &[u_min_relaxed, u_max_relaxed] = relaxed_bounds;

      /* Apply a stricter window: */

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();

      const Number u_relaxation =
          ScalarNumber(2. * limiter_.relaxation_factor()) *
          std::abs(u_relaxation_numerator_) /
          (std::abs(u_relaxation_denominator_) + Number(eps));

      u_min_relaxed = std::max(u_min_relaxed, u_min - u_relaxation);
      u_max_relaxed = std::min(u_max_relaxed, u_max + u_relaxation);

      return relaxed_bounds;
    }
  } // namespace ScalarConservation
} // namespace ryujin
