//
// SPDX-License-Identifier: Apache-2.0
// [LANL Copyright Statement]
// Copyright (C) 2023 - 2026 by the ryujin authors
// Copyright (C) 2023 - 2024 by Triad National Security, LLC
//

#pragma once

#include <compile_time_options.h>

#include "hyperbolic_system.h"

#include <multicomponent_vector.h>
#include <newton.h>
#include <observer_pointer.h>
#include <simd.h>

// #define DEBUG_OUTPUT_LIMITER

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

      template <int, typename>
      friend class LimiterView;
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
       * Return the number of limiter iterations.
       */
      unsigned int iterations() const
      {
        return limiter_.iterations_;
      }

      /**
       * Return the tolerance for the quadratic Newton stopping criterion.
       */
      ScalarNumber newton_tolerance() const
      {
        return limiter_.newton_tolerance_;
      }

      /**
       * Return the maximal number of quadratic Newton iterations.
       */
      unsigned int newton_max_iterations() const
      {
        return limiter_.newton_max_iterations_;
      }

      /**
       * Return the factor used for scaling the relaxation window.
       */
      ScalarNumber relaxation_factor() const
      {
        return limiter_.relaxation_factor_;
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
      r *= relaxation_factor();

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

      const Number h_relaxed = ScalarNumber(2. * relaxation_factor()) *
                               std::abs(h_relaxation_numerator_) /
                               (relaxation_denominator_ + Number(eps));

      const Number v2_relaxed = ScalarNumber(2. * relaxation_factor()) *
                                std::abs(v2_relaxation_numerator_) /
                                (relaxation_denominator_ + Number(eps));

      h_min_relaxed = std::max(h_min_relaxed, h_min - h_relaxed);
      h_max_relaxed = std::min(h_max_relaxed, h_max + h_relaxed);
      v2_max_relaxed = std::min(v2_max_relaxed, v2_max + v2_relaxed);

      return relaxed_bounds;
    }


    template <int dim, typename Number>
    inline std::tuple<Number, bool>
    LimiterView<dim, Number>::limit(const Bounds &bounds,
                                    const state_type &U,
                                    const state_type &P,
                                    const Number t_min /* = Number(0.) */,
                                    const Number t_max /* = Number(1.) */) const
    {
      bool success = true;
      Number t_l = t_min;
      Number t_r = t_max;

      const auto &[h_min, h_max, v2_max] = bounds;

      constexpr ScalarNumber min = std::numeric_limits<ScalarNumber>::min();
      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
      const auto small = view_.dry_state_relaxation_small();
      const auto large = view_.dry_state_relaxation_large();
      const auto relax_small = ScalarNumber(1. + small * eps);
      const auto relax = ScalarNumber(1. + large * eps);

      /*
       * We first limit the water_depth h.
       *
       * See [Guermond et al, 2021] (5.7).
       */

      {
        auto h_U = view_.water_depth(U);
        const auto &h_P = view_.water_depth(P);

        const auto test_min = view_.filter_dry_water_depth(
            std::max(Number(0.), h_U - relax * h_max));
        const auto test_max = view_.filter_dry_water_depth(
            std::max(Number(0.), h_min - relax * h_U));

        if (!(test_min == Number(0.) && test_max == Number(0.))) {
#ifdef DEBUG_OUTPUT
          std::cout << std::fixed << std::setprecision(16);
          std::cout << "Bounds violation: low-order water depth (critical)!\n"
                    << "\n\t\th min:         " << h_min
                    << "\n\t\th min (delta): " << negative_part(h_U - h_min)
                    << "\n\t\th:             " << h_U
                    << "\n\t\th max (delta): " << positive_part(h_U - h_max)
                    << "\n\t\th max:         " << h_max << "\n"
                    << std::endl;
#endif
          success = false;
        }

        const Number denominator =
            ScalarNumber(1.) / (std::abs(h_P) + eps * h_max + min);

        constexpr auto lt = dealii::SIMDComparison::less_than;

        t_r = dealii::compare_and_apply_mask<lt>( //
            h_max,
            h_U + t_r * h_P,
            /*
             * h_P is positive.
             *
             * Note: Do not take an absolute value here. If we are out of
             * bounds we have to ensure that t_r is set to t_min.
             */
            (h_max - h_U) * denominator,
            t_r);

        t_r = dealii::compare_and_apply_mask<lt>( //
            h_U + t_r * h_P,
            h_min,
            /*
             * h_P is negative.
             *
             * Note: Do not take an absolute value here. If we are out of
             * bounds we have to ensure that t_r is set to t_min.
             */
            (h_U - h_min) * denominator,
            t_r);

        /*
         * Ensure that t_min <= t <= t_max. This might not be the case if
         * h_U is outside the interval [h_min, h_max]. Furthermore, the
         * quotient we take above is prone to numerical cancellation in
         * particular in the second pass of the limiter when h_P might be
         * small.
         */
        t_r = std::min(t_r, t_max);
        t_r = std::max(t_r, t_min);


#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
        /*
         * Verify that the new state is within bounds:
         */
        const auto h_new = view_.water_depth(U + t_r * P);
        const auto test_new_min = view_.filter_dry_water_depth(
            std::max(Number(0.), h_new - relax * h_max));
        const auto test_new_max = view_.filter_dry_water_depth(
            std::max(Number(0.), h_min - relax * h_new));

        if (!(test_new_min == Number(0.) && test_new_max == Number(0.))) {
#ifdef DEBUG_OUTPUT
          std::cout << std::fixed << std::setprecision(30);
          std::cout << "Bounds violation: high-order water depth!\n"
                    << "\n\t\th min:         " << h_min
                    << "\n\t\th min (delta): " << negative_part(h_new - h_min)
                    << "\n\t\th:             " << h_new
                    << "\n\t\th max (delta): " << positive_part(h_new - h_max)
                    << "\n\t\th max:         " << h_max << "\n"
                    << std::endl;
#endif
          success = false;
        }
#endif
      }

      /*
       * Limit the (negative) |v|^2:
       *
       * Given initial limiter values t_l and t_r with psi(t_l) > 0 and
       * psi(t_r) < 0 we try to find t^\ast with psi(t^\ast) \approx 0.
       *
       * Here, psi is the function:
       *
       *   psi = h^2 (|v|^2)^max - |q|^2
       */

      {
        /* We first check if t_r is a good state */

        const auto U_r = U + t_r * P;
        const auto h_r = view_.water_depth(U_r);
        const auto q_r = view_.momentum(U_r);

        const auto psi_r = relax_small * h_r * h_r * v2_max - q_r.norm_square();

        /*
         * If psi_r > 0 the right state is fine, force returning t_r by
         * setting t_l = t_r:
         */
        t_l = dealii::compare_and_apply_mask<
            dealii::SIMDComparison::greater_than>(psi_r, Number(0.), t_r, t_l);

        /* If we have set t_l = t_r everywhere we can return: */
        if (t_l == t_r)
          return {t_l, success};

#ifdef DEBUG_OUTPUT_LIMITER
        {
          std::cout << std::endl;
          std::cout << std::fixed << std::setprecision(16);
          std::cout << "t_l: (start) " << t_l << std::endl;
          std::cout << "t_r: (start) " << t_r << std::endl;
        }
#endif

        const auto U_l = U + t_l * P;
        const auto h_l = view_.water_depth(U_l);
        const auto q_l = view_.momentum(U_l);

        const auto psi_l = relax_small * h_l * h_l * v2_max - q_l.norm_square();

        /*
         * Verify that the left state is within bounds. This property might
         * be violated for relative CFL numbers larger than 1.
         *
         * We use a non-scaled eps here to force the lower_bound to be
         * negative so that we do not accidentally trigger in "perfect" dry
         * states with h_l equal to zero.
         */
        const auto filtered_h_l = view_.filter_dry_water_depth(h_l);
        const auto lower_bound =
            (ScalarNumber(1.) - relax) * filtered_h_l * filtered_h_l * v2_max -
            ScalarNumber(100.) * eps;
        if (!(std::min(Number(0.), psi_l - lower_bound) == Number(0.))) {
#ifdef DEBUG_OUTPUT
          std::cout << std::fixed << std::setprecision(16);
          std::cout
              << "Bounds violation: low-order square velocity (critical)!\n";
          std::cout << "\t\tPsi left: 0 <= " << psi_l << "\n" << std::endl;
#endif
          success = false;
        }

        /*
         * Skip the quadratic Newton step if the window between t_l and t_r
         * is within the prescribed tolerance:
         */
        const Number tolerance(newton_tolerance());
        if (!(std::max(Number(0.), t_r - t_l - tolerance) == Number(0.))) {
          /*
           * If the bound is not satisfied, we need to find the root of a
           * quadratic function:
           *
           * psi(t)   = r (h_U + t h_P)^2 v2_max
           *            - (|q_U|^2 + 2(q_U * q_P) t + |q_P|^2 t^2)
           *
           * d_psi(t) = 2 r (h_U + t * h_P) * h_P v2_max
           *            - 2 (q_U * q_P) - 2 |q_P|^2 t
           *
           * where r = relax_small.
           *
           * We can compute the root of this function efficiently by using our
           * standard quadratic_newton_step() function that will use the points
           * [p1, p1, p2] as well as [p1, p2, p2] to construct two quadratic
           * polynomials to compute new candiates for the bounds [t_l, t_r]. In
           * case of a quadratic function psi(t) both polynomials will coincide
           * so that (up to round-off error) t_l = t_r.
           */
          const auto &h_U = view_.water_depth(U);
          const auto &h_P = view_.water_depth(P);
          const auto &q_U = view_.momentum(U);
          const auto &q_P = view_.momentum(P);

          const auto dpsi_l = ScalarNumber(2.) *
                              (relax_small * (h_U + t_l * h_P) * h_P * v2_max -
                               ((q_U * q_P) + q_P * q_P * t_l));
          const auto dpsi_r = ScalarNumber(2.) *
                              (relax_small * (h_U + t_r * h_P) * h_P * v2_max -
                               ((q_U * q_P) + q_P * q_P * t_r));

          quadratic_newton_step(
              t_l, t_r, psi_l, psi_r, dpsi_l, dpsi_r, Number(-1.));

#ifdef DEBUG_OUTPUT_LIMITER
          if (std::max(Number(0.), psi_r + Number(eps)) == Number(0.)) {
            std::cout << "psi_l:       " << psi_l << std::endl;
            std::cout << "psi_r:       " << psi_r << std::endl;
            std::cout << "dpsi_l:      " << dpsi_l << std::endl;
            std::cout << "dpsi_r:      " << dpsi_r << std::endl;
            std::cout << "t_l: (end)   " << t_l << std::endl;
            std::cout << "t_r: (end)   " << t_r << std::endl;
          }
#endif
        }

#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
        /*
         * Verify that the new state is within bounds:
         */
        {
          const auto U_new = U + t_l * P;
          const auto h_new = view_.water_depth(U_new);
          const auto q_new = view_.momentum(U_new);

          const auto psi_new =
              relax_small * h_new * h_new * v2_max - q_new.norm_square();

          const auto lower_bound =
              (ScalarNumber(1.) - relax) * h_new * h_new * v2_max -
              ScalarNumber(100.) * eps;

          const bool psi_valid =
              std::min(Number(0.), psi_new - lower_bound) == Number(0.);
          if (!psi_valid) {
#ifdef DEBUG_OUTPUT
            std::cout << std::fixed << std::setprecision(16);
            std::cout << "Bounds violation: high-order square velocity!\n";
            std::cout << "\t\tPsi: 0 <= " << psi_new << "\n" << std::endl;
#endif
            success = false;
          }
        }
#endif
      }

      return {t_l, success};
    }
  } // namespace ShallowWater
} // namespace ryujin
