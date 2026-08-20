//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "hyperbolic_system.h"

#include <compile_time_options.h>
#include <deal.II/base/config.h>
#include <deal.II/base/memory_space.h>
#include <multicomponent_vector.h>
#include <newton.h>
#include <observer_pointer.h>
#include <simd.h>

// #define DEBUG_OUTPUT_LIMITER

namespace ryujin
{
  namespace Euler
  {
    template <int dim,
              typename Number = double,
              typename MemorySpace = dealii::MemorySpace::Host>
    class LimiterView;

    /**
     * The convex limiter.
     *
     * The limiter implements a convex limiting technique as described in
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
    template <typename ScalarNumber = double>
    class Limiter : public dealii::ParameterAcceptor
    {
    public:
      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      /**
       * Alias for the view on the limiter for a given dimension @p dim,
       * choice of number type @p Number, and memory space @p MemorySpace.
       */
      template <int dim,
                typename Number = double,
                typename MemorySpace = dealii::MemorySpace::Host>
      using View = LimiterView<dim, Number, MemorySpace>;

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
       * double, as well as a VectorizedArray holding packed scalars). The
       * optional @p MemorySpace template parameter selects whether the
       * view is intended for the host or device memory space.
       */
      template <int dim,
                typename Number,
                typename MemorySpace = dealii::MemorySpace::Host>
      auto view() const
      {
        return View<dim, Number, MemorySpace>{
            hyperbolic_system_->template view<dim, Number, MemorySpace>(),
            *this};
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
     * @ingroup EulerEquations
     */
    template <int dim, typename Number, typename MemorySpace>
    class LimiterView
    {
    public:
      static_assert(
          std::is_same_v<MemorySpace, dealii::MemorySpace::Host> ||
              std::is_same_v<MemorySpace, dealii::MemorySpace::Default>,
          "Unexpected memory space");

      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      using View = HyperbolicSystemView<dim, Number, MemorySpace>;

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
          , newton_tolerance_(limiter.newton_tolerance())
          , newton_max_iterations_(limiter.newton_max_iterations())
          , relaxation_factor_(limiter.relaxation_factor())
      {
      }

      /**
       * Given a state @p U_i and an index @p i return "strict" bounds,
       * i.e., a minimal convex set containing the state.
       */
      DEAL_II_HOST_DEVICE Bounds
      projection_bounds_from_state(const PrecomputedVectorView &pv,
                                   const unsigned int i,
                                   const state_type &U_i) const;

      /**
       * Given two bounds bounds_left, bounds_right, this function computes
       * a larger, combined set of bounds that this is a (convex) superset
       * of the two.
       */
      DEAL_II_HOST_DEVICE Bounds
      combine_bounds(const Bounds &bounds_left,
                     const Bounds &bounds_right) const;

      /**
       * This function applies a relaxation to a given a (strict) bound @p
       * bounds using a non dimensionalized measure @p hd (that should
       * scale as $h^d$, where $h$ is the local mesh size). This is done
       * for the case of the Euler equations by multiplying maximum bounds
       * with $(1+r)$ and minimum bounds with $(1-r)$, while ensuring that
       * the bounds still describe an admissible state.
       */
      DEAL_II_HOST_DEVICE Bounds fully_relax_bounds(const Bounds &bounds,
                                                    const Number &hd) const;

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
      DEAL_II_HOST_DEVICE void reset(const PrecomputedVectorView &pv,
                                     const unsigned int i,
                                     const state_type &U_i,
                                     const flux_contribution_type &flux_i);

      /**
       * When looping over the sparsity row, add the contribution associated
       * with the neighboring state U_j.
       */
      DEAL_II_HOST_DEVICE void
      accumulate(const PrecomputedVectorView &pv,
                 const unsigned int *js,
                 const state_type &U_j,
                 const flux_contribution_type &flux_j,
                 const dealii::Tensor<1, dim, Number> &scaled_c_ij,
                 const state_type &affine_shift);

      /**
       * Return the computed bounds (with relaxation applied).
       */
      DEAL_II_HOST_DEVICE Bounds bounds(const Number hd_i) const;

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
      DEAL_II_HOST_DEVICE std::tuple<Number, bool>
      limit(const Bounds &bounds,
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
      ScalarNumber newton_tolerance_;
      unsigned int newton_max_iterations_;
      ScalarNumber relaxation_factor_;

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


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    LimiterView<dim, Number, MemorySpace>::projection_bounds_from_state(
        const PrecomputedVectorView &pv,
        const unsigned int i,
        const state_type &U_i) const -> Bounds
    {
      const auto rho_i = view_.density(U_i);
      const auto &[s_i, eta_i] =
          pv.template read_tensor<Number, precomputed_type>(i);

      return {/*rho_min*/ rho_i, /*rho_max*/ rho_i, /*s_min*/ s_i};
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    LimiterView<dim, Number, MemorySpace>::combine_bounds(
        const Bounds &bounds_left, const Bounds &bounds_right) const -> Bounds
    {
      const auto &[rho_min_l, rho_max_l, s_min_l] = bounds_left;
      const auto &[rho_min_r, rho_max_r, s_min_r] = bounds_right;

      return {std::min(rho_min_l, rho_min_r),
              std::max(rho_max_l, rho_max_r),
              std::min(s_min_l, s_min_r)};
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    LimiterView<dim, Number, MemorySpace>::fully_relax_bounds(
        const Bounds &bounds, const Number &hd) const -> Bounds
    {
      auto relaxed_bounds = bounds;
      auto &[rho_min, rho_max, s_min] = relaxed_bounds;

      /* Use r = factor * (m_i / |Omega|) ^ (1.5 / d): */

      Number r = std::sqrt(hd);                              // in 3D: ^ 3/6
      if constexpr (dim == 2)                                //
        r = dealii::Utilities::fixed_power<3>(std::sqrt(r)); // in 2D: ^ 3/4
      else if constexpr (dim == 1)                           //
        r = dealii::Utilities::fixed_power<3>(r);            // in 1D: ^ 3/2
      r *= relaxation_factor_;

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
      rho_min *= std::max(Number(1.) - r, Number(eps));
      rho_max *= (Number(1.) + r);
      s_min *= std::max(Number(1.) - r, Number(eps));

      return relaxed_bounds;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    LimiterView<dim, Number, MemorySpace>::reset(
        const PrecomputedVectorView & /*pv*/,
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


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    LimiterView<dim, Number, MemorySpace>::accumulate(
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


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    LimiterView<dim, Number, MemorySpace>::bounds(const Number hd_i) const
        -> Bounds
    {
      const auto &[rho_min, rho_max, s_min] = bounds_;

      auto relaxed_bounds = fully_relax_bounds(bounds_, hd_i);
      auto &[rho_min_relaxed, rho_max_relaxed, s_min_relaxed] = relaxed_bounds;

      /* Apply a stricter window: */

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();

      const auto rho_relaxation =
          ScalarNumber(2. * relaxation_factor_) *
          std::abs(rho_relaxation_numerator_) /
          (std::abs(rho_relaxation_denominator_) + Number(eps));

      const auto entropy_relaxation =
          relaxation_factor_ * (s_interp_max_ - s_min);

      rho_min_relaxed = std::max(rho_min_relaxed, rho_min - rho_relaxation);
      rho_max_relaxed = std::min(rho_max_relaxed, rho_max + rho_relaxation);
      s_min_relaxed = std::max(s_min_relaxed, s_min - entropy_relaxation);

      return relaxed_bounds;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE std::tuple<Number, bool>
    LimiterView<dim, Number, MemorySpace>::limit(
        const Bounds &bounds,
        const state_type &U,
        const state_type &P,
        const Number t_min /* = Number(0.) */,
        const Number t_max /* = Number(1.) */) const
    {
      bool success = true;
      Number t_r = t_max;

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
      const auto small = view_.vacuum_state_relaxation_small();
      const auto large = view_.vacuum_state_relaxation_large();
      const ScalarNumber relax_small = ScalarNumber(1. + small * eps);
      const ScalarNumber relax = ScalarNumber(1. + large * eps);

      /*
       * First limit the density rho.
       *
       * See [Guermond, Nazarov, Popov, Thomas] (4.8):
       */

      {
        const auto &rho_U = view_.density(U);
        const auto &rho_P = view_.density(P);

        const auto &rho_min = std::get<0>(bounds);
        const auto &rho_max = std::get<1>(bounds);

        /*
         * Verify that rho_U is within bounds. This property might be
         * violated for relative CFL numbers larger than 1.
         */
        const auto test_min = view_.filter_vacuum_density(
            std::max(Number(0.), rho_U - relax * rho_max));
        const auto test_max = view_.filter_vacuum_density(
            std::max(Number(0.), rho_min - relax * rho_U));
        if (!(test_min == Number(0.) && test_max == Number(0.))) {
#ifdef DEBUG_OUTPUT
          std::cout << std::fixed << std::setprecision(16);
          std::cout << "Bounds violation: low-order density (critical)!"
                    << "\n\t\trho min:         " << rho_min
                    << "\n\t\trho min (delta): "
                    << negative_part(rho_U - rho_min)
                    << "\n\t\trho:             " << rho_U
                    << "\n\t\trho max (delta): "
                    << positive_part(rho_U - rho_max)
                    << "\n\t\trho max:         " << rho_max << "\n"
                    << std::endl;
#endif
          success = false;
        }

        const Number denominator =
            ScalarNumber(1.) / (std::abs(rho_P) + eps * rho_max);

        constexpr auto lt = dealii::SIMDComparison::less_than;

        t_r = ryujin::compare_and_apply_mask<lt>( //
            rho_max,
            rho_U + t_r * rho_P,
            /*
             * rho_P is positive.
             *
             * Note: Do not take an absolute value here. If we are out of
             * bounds we have to ensure that t_r is set to t_min.
             */
            (rho_max - rho_U) * denominator,
            t_r);

        t_r = ryujin::compare_and_apply_mask<lt>( //
            rho_U + t_r * rho_P,
            rho_min,
            /*
             * rho_P is negative.
             *
             * Note: Do not take an absolute value here. If we are out of
             * bounds we have to ensure that t_r is set to t_min.
             */
            (rho_U - rho_min) * denominator,
            t_r);

        /*
         * Ensure that t_min <= t <= t_max. This might not be the case if
         * rho_U is outside the interval [rho_min, rho_max]. Furthermore,
         * the quotient we take above is prone to numerical cancellation in
         * particular in the second pass of the limiter when rho_P might be
         * small.
         */
        t_r = std::min(t_r, t_max);
        t_r = std::max(t_r, t_min);

#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
        /*
         * Verify that the new state is within bounds:
         */
        const auto rho_new = view_.density(U + t_r * P);
        const auto test_new_min = view_.filter_vacuum_density(
            std::max(Number(0.), rho_new - relax * rho_max));
        const auto test_new_max = view_.filter_vacuum_density(
            std::max(Number(0.), rho_min - relax * rho_new));
        if (!(test_new_min == Number(0.) && test_new_max == Number(0.))) {
#ifdef DEBUG_OUTPUT
          std::cout << std::fixed << std::setprecision(16);
          std::cout << "Bounds violation: high-order density!"
                    << "\n\t\trho min:         " << rho_min
                    << "\n\t\trho min (delta): "
                    << negative_part(rho_new - rho_min)
                    << "\n\t\trho:             " << rho_new
                    << "\n\t\trho max (delta): "
                    << positive_part(rho_new - rho_max)
                    << "\n\t\trho max:         " << rho_max << "\n"
                    << std::endl;
#endif
          success = false;
        }
#endif
      }

      /*
       * Then limit the specific entropy:
       *
       * See [Guermond, Nazarov, Popov, Thomas], Section 4.6 + Section 5.1:
       */

      Number t_l = t_min; // good state

      const ScalarNumber gamma = view_.gamma();
      const ScalarNumber gp1 = gamma + ScalarNumber(1.);

      {
        /*
         * Prepare a quadratic Newton method:
         *
         * Given initial limiter values t_l and t_r with psi(t_l) > 0 and
         * psi(t_r) < 0 we try to find t^\ast with psi(t^\ast) \approx 0.
         *
         * Here, psi is a 3-convex function obtained by scaling the specific
         * entropy s:
         *
         *   psi = \rho ^ {\gamma + 1} s
         *
         * (s in turn was defined as s =\varepsilon \rho ^{-\gamma}, where
         * \varepsilon = (\rho e) is the internal energy.)
         */

        const auto &s_min = std::get<2>(bounds);

#ifdef DEBUG_OUTPUT_LIMITER
        std::cout << std::endl;
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "t_l: (start) " << t_l << std::endl;
        std::cout << "t_r: (start) " << t_r << std::endl;
#endif

        for (unsigned int n = 0; n < newton_max_iterations_; ++n) {

          const auto U_r = U + t_r * P;
          const auto rho_r = view_.density(U_r);
          const auto rho_r_gamma = ryujin::pow(rho_r, gamma);
          const auto rho_e_r = view_.internal_energy(U_r);

          auto psi_r =
              relax_small * rho_r * rho_e_r - s_min * rho_r * rho_r_gamma;

#ifndef DEBUG_EXPENSIVE_BOUNDS_CHECK
          /*
           * If psi_r > 0 the right state is fine, force returning t_r by
           * setting t_l = t_r:
           */
          t_l = ryujin::compare_and_apply_mask<
              dealii::SIMDComparison::greater_than>(
              psi_r, Number(0.), t_r, t_l);

          /*
           * If we have set t_l = t_r everywhere then all states state U_r
           * with t_r obey the specific entropy inequality and we can
           * break.
           *
           * This is a very important optimization: Only for 1 in (25 to
           * 50) cases do we actually need to limit on the specific entropy
           * because one of the right states failed. So we can skip
           * constructing the left state U_l, which is expensive.
           *
           * This implies unfortunately that we might not accurately report
           * whether the low_order update U itself obeyed bounds because
           * U_r = U + t_r * P pushed us back into bounds. We thus skip
           * this shortcut if `DEBUG_EXPENSIVE_BOUNDS_CHECK` is set.
           */
          if (t_l == t_r) {
#ifdef DEBUG_OUTPUT_LIMITER
            std::cout << "shortcut: t_l == t_r" << std::endl;
            std::cout << "psi_l:       " << psi_l << std::endl;
            std::cout << "psi_r:       " << psi_r << std::endl;
            std::cout << "t_l: (  " << n << "  ) " << t_l << std::endl;
            std::cout << "t_r: (  " << n << "  ) " << t_r << std::endl;
#endif
            break;
          }
#endif

          const auto U_l = U + t_l * P;
          const auto rho_l = view_.density(U_l);
          const auto rho_l_gamma = ryujin::pow(rho_l, gamma);
          const auto rho_e_l = view_.internal_energy(U_l);

          auto psi_l =
              relax_small * rho_l * rho_e_l - s_min * rho_l * rho_l_gamma;

          /*
           * Verify that the left state is within bounds. This property might
           * be violated for relative CFL numbers larger than 1.
           */
          const auto lower_bound =
              (ScalarNumber(1.) - relax) * s_min * rho_l * rho_l_gamma;
          if (n == 0 &&
              !(std::min(Number(0.), psi_l - lower_bound) == Number(0.))) {
#ifdef DEBUG_OUTPUT
            std::cout << std::fixed << std::setprecision(16);
            std::cout
                << "Bounds violation: low-order specific entropy (critical)!\n";
            std::cout << "\t\tPsi left: 0 <= " << psi_l << "\n" << std::endl;
#endif
            success = false;
          }

#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
          /*
           * If psi_r > 0 the right state is fine, force returning t_r by
           * setting t_l = t_r:
           */
          t_l = ryujin::compare_and_apply_mask<
              dealii::SIMDComparison::greater_than>(
              psi_r, Number(0.), t_r, t_l);
#endif

          /*
           * Break if the window between t_l and t_r is within the prescribed
           * tolerance:
           */
          const Number tolerance(newton_tolerance_);
          if (std::max(Number(0.), t_r - t_l - tolerance) == Number(0.)) {
#ifdef DEBUG_OUTPUT_LIMITER
            std::cout << "break: t_l and t_r within tolerance" << std::endl;
            std::cout << "psi_l:       " << psi_l << std::endl;
            std::cout << "psi_r:       " << psi_r << std::endl;
            std::cout << "t_l: (  " << n << "  ) " << t_l << std::endl;
            std::cout << "t_r: (  " << n << "  ) " << t_r << std::endl;
#endif
            break;
          }

          /* We got unlucky and have to perform a Newton step: */

          const auto drho = view_.density(P);
          const auto drho_e_l = view_.internal_energy_derivative(U_l) * P;
          const auto drho_e_r = view_.internal_energy_derivative(U_r) * P;
          const auto dpsi_l =
              rho_l * drho_e_l + (rho_e_l - gp1 * s_min * rho_l_gamma) * drho;
          const auto dpsi_r =
              rho_r * drho_e_r + (rho_e_r - gp1 * s_min * rho_r_gamma) * drho;

          quadratic_newton_step(
              t_l, t_r, psi_l, psi_r, dpsi_l, dpsi_r, Number(-1.));

#ifdef DEBUG_OUTPUT_LIMITER
          std::cout << "psi_l:       " << psi_l << std::endl;
          std::cout << "psi_r:       " << psi_r << std::endl;
          std::cout << "dpsi_l:      " << dpsi_l << std::endl;
          std::cout << "dpsi_r:      " << dpsi_r << std::endl;
          std::cout << "t_l: (  " << n << "  ) " << t_l << std::endl;
          std::cout << "t_r: (  " << n << "  ) " << t_r << std::endl;
#endif
        }

#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
        /*
         * Verify that the new state is within bounds:
         */
        {
          const auto U_new = U + t_l * P;
          const auto rho_new = view_.density(U_new);
          const auto rho_new_gamma = ryujin::pow(rho_new, gamma);
          const auto rho_e_new = view_.internal_energy(U_new);

          auto psi_new = relax_small * rho_new * rho_e_new -
                         s_min * rho_new * rho_new_gamma;

          const auto lower_bound =
              (ScalarNumber(1.) - relax) * s_min * rho_new * rho_new_gamma;

          const bool e_valid = std::min(Number(0.), rho_e_new) == Number(0.);
          const bool psi_valid =
              std::min(Number(0.), psi_new - lower_bound) == Number(0.);

          if (!e_valid || !psi_valid) {
#ifdef DEBUG_OUTPUT
            std::cout << std::fixed << std::setprecision(16);
            std::cout << "Bounds violation: high-order specific entropy!\n";
            std::cout << "\t\trho e: 0 <= " << rho_e_new << "\n";
            std::cout << "\t\tPsi:   0 <= " << psi_new << "\n" << std::endl;
#endif
            success = false;
          }
        }
#endif
      }

      return {t_l, success};
    }
  } // namespace Euler
} // namespace ryujin
