//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "hyperbolic_system.h"

#include <compile_time_options.h>
#include <deal.II/base/config.h>
#include <multicomponent_vector.h>
#include <newton.h>
#include <observer_pointer.h>
#include <simd.h>

namespace ryujin
{
  namespace Euler
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

        if constexpr (std::is_same<ScalarNumber, double>::value)
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

      ACCESSOR_READ_ONLY(iterations);
      ACCESSOR_READ_ONLY(newton_tolerance);
      ACCESSOR_READ_ONLY(newton_max_iterations);
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
      ScalarNumber newton_tolerance_;
      unsigned int newton_max_iterations_;
      ScalarNumber relaxation_factor_;
    };


    /**
     * The convex limiter.
     *
     * The class implements a convex limiting technique as described in
     * @cite GuermondEtAl2018 and @cite ryujin-2021-1. Given a
     * computed set of bounds and an update direction \f$\mathbf P_{ij}\f$
     * one can now determine a candidate \f$\tilde l_{ij}\f$ by computing
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
       * for the case of the Euler equations by multiplying maximum bounds
       * with $(1+r)$ and minimum bounds with $(1-r)$, while ensuring that
       * the bounds still describe an admissible state.
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
       *
       * The returned boolean is set to true if the original low-order
       * update was within bounds.
       *
       * @note If the debug option `DEBUG_EXPENSIVE_BOUNDS_CHECK` is set to
       * true, then the boolean is set to true if the low-order and the
       * resulting high-order update are within bounds. The latter might be
       * violated due to round-off errors when computing the limiter
       * bounds.
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

      Number rho_relaxation_numerator_;
      Number rho_relaxation_denominator_;
      Number s_interp_max_;

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
        const PrecomputedVectorView &pv,
        const unsigned int i,
        const state_type &U_i) const -> Bounds
    {
      const auto rho_i = view_.density(U_i);
      const auto &[s_i, eta_i] =
          pv.template read_tensor<Number, precomputed_type>(i);

      return {/*rho_min*/ rho_i, /*rho_max*/ rho_i, /*s_min*/ s_i};
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto LimiterView<dim, Number>::combine_bounds(
        const Bounds &bounds_left, const Bounds &bounds_right) const -> Bounds
    {
      const auto &[rho_min_l, rho_max_l, s_min_l] = bounds_left;
      const auto &[rho_min_r, rho_max_r, s_min_r] = bounds_right;

      return {std::min(rho_min_l, rho_min_r),
              std::max(rho_max_l, rho_max_r),
              std::min(s_min_l, s_min_r)};
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    LimiterView<dim, Number>::fully_relax_bounds(const Bounds &bounds,
                                                 const Number &hd) const
        -> Bounds
    {
      auto relaxed_bounds = bounds;
      auto &[rho_min, rho_max, s_min] = relaxed_bounds;

      /* Use r = factor * (m_i / |Omega|) ^ (1.5 / d): */

      Number r = std::sqrt(hd);                              // in 3D: ^ 3/6
      if constexpr (dim == 2)                                //
        r = dealii::Utilities::fixed_power<3>(std::sqrt(r)); // in 2D: ^ 3/4
      else if constexpr (dim == 1)                           //
        r = dealii::Utilities::fixed_power<3>(r);            // in 1D: ^ 3/2
      r *= limiter_.relaxation_factor();

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
      rho_min *= std::max(Number(1.) - r, Number(eps));
      rho_max *= (Number(1.) + r);
      s_min *= std::max(Number(1.) - r, Number(eps));

      return relaxed_bounds;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void
    LimiterView<dim, Number>::reset(const PrecomputedVectorView & /*pv*/,
                                    const unsigned int /*i*/,
                                    const state_type &U_i,
                                    const flux_contribution_type & /*flux_i*/)
    {
      U_i_ = U_i;

      /* Bounds: */

      auto &[rho_min, rho_max, s_min] = bounds_;

      rho_min = Number(std::numeric_limits<ScalarNumber>::max());
      rho_max = Number(0.);
      s_min = Number(std::numeric_limits<ScalarNumber>::max());

      /* Relaxation: */

      rho_relaxation_numerator_ = Number(0.);
      rho_relaxation_denominator_ = Number(0.);
      s_interp_max_ = Number(0.);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void LimiterView<dim, Number>::accumulate(
        const PrecomputedVectorView &pv,
        const unsigned int *js,
        const state_type &U_j,
        const flux_contribution_type & /*flux_j*/,
        const dealii::Tensor<1, dim, Number> &scaled_c_ij,
        const state_type &affine_shift)
    {
      // TODO: Currently we only apply the affine_shift to U_ij_bar (which
      // then enters all bounds), but we do not modify s_interp and
      // rho_relaxation. When actually adding a source term to the Euler
      // equations verify that this does the right thing.
      Assert(std::max(affine_shift.norm(), Number(0.)) == Number(0.),
             dealii::ExcNotImplemented());

      /* Bounds: */
      auto &[rho_min, rho_max, s_min] = bounds_;

      const auto rho_i = view_.density(U_i_);
      const auto m_i = view_.momentum(U_i_);
      const auto rho_j = view_.density(U_j);
      const auto m_j = view_.momentum(U_j);
      const auto rho_affine_shift = view_.density(affine_shift);

      /* bar state shifted by an affine shift: */
      const auto rho_ij_bar =
          ScalarNumber(0.5) * (rho_i + rho_j + (m_i - m_j) * scaled_c_ij) +
          rho_affine_shift;

      rho_min = std::min(rho_min, rho_ij_bar);
      rho_max = std::max(rho_max, rho_ij_bar);

      const auto &[s_j, eta_j] =
          pv.template read_tensor<Number, precomputed_type>(js);
      s_min = std::min(s_min, s_j);

      /* Relaxation: */

      /* Use a uniform weight. */
      const auto beta_ij = Number(1.);
      rho_relaxation_numerator_ += beta_ij * (rho_i + rho_j);
      rho_relaxation_denominator_ += std::abs(beta_ij);

      const Number s_interp =
          view_.specific_entropy((U_i_ + U_j) * ScalarNumber(.5));
      s_interp_max_ = std::max(s_interp_max_, s_interp);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    LimiterView<dim, Number>::bounds(const Number hd_i) const -> Bounds
    {
      const auto &[rho_min, rho_max, s_min] = bounds_;

      auto relaxed_bounds = fully_relax_bounds(bounds_, hd_i);
      auto &[rho_min_relaxed, rho_max_relaxed, s_min_relaxed] = relaxed_bounds;

      /* Apply a stricter window: */

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();

      const auto rho_relaxation =
          ScalarNumber(2. * limiter_.relaxation_factor()) *
          std::abs(rho_relaxation_numerator_) /
          (std::abs(rho_relaxation_denominator_) + Number(eps));

      const auto entropy_relaxation =
          limiter_.relaxation_factor() * (s_interp_max_ - s_min);

      rho_min_relaxed = std::max(rho_min_relaxed, rho_min - rho_relaxation);
      rho_max_relaxed = std::min(rho_max_relaxed, rho_max + rho_relaxation);
      s_min_relaxed = std::max(s_min_relaxed, s_min - entropy_relaxation);

      return relaxed_bounds;
    }
  } // namespace Euler
} // namespace ryujin
