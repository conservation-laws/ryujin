//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "hyperbolic_system.h"

#include <gpu.h>
#include <multicomponent_vector.h>
#include <observer_pointer.h>
#include <simd.h>

namespace ryujin
{
  namespace ScalarConservation
  {
    template <int dim,
              typename Number = double,
              typename MemorySpace = dealii::MemorySpace::Host>
    class LimiterView;

    /**
     * The convex limiter.
     *
     * @ingroup ScalarConservationEquations
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
       * A structure holding all runtime parameters of the limiter.
       */
      struct Parameters {
        unsigned int iterations;
        double relaxation_factor;
      };

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
          , parameters_("scalar_conservation_limiter_parameters",
                        TransferPolicy::implicit_transfers_host_resident)
          , hyperbolic_system_(&hyperbolic_system)
      {
        /* reference remains valid due to implicit_transfers_host_resident */
        auto &parameters = *parameters_.view();

        parameters.iterations = 2;
        add_parameter("iterations",
                      parameters.iterations,
                      "Number of limiter iterations");

        parameters.relaxation_factor = 1.;
        add_parameter("relaxation factor",
                      parameters.relaxation_factor,
                      "Factor for scaling the relaxation window with r_i = "
                      "factor * (m_i/|Omega|)^(1.5/d).");

        /* invalidates view on default memory space */
        ParameterAcceptor::parse_parameters_call_back.connect(
            [this] { parameters_.view(); });
      }

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
        return LimiterView<dim, Number, MemorySpace>{
            hyperbolic_system_->template view<dim, Number, MemorySpace>(),
            *this};
      }

    private:
      //@}
      /**
       * @name Run time options
       */
      //@{

      Mirrored<Parameters> parameters_;

      //@}
      /**
       * @name Internal data
       */
      //@{

      dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;

      template <int, typename, typename>
      friend class LimiterView;

      //@}
    };


    /**
     * A view of the Limiter that makes the interface available for a given
     * dimension @p dim and choice of number type @p Number (which can be a
     * scalar float, or double, as well as a VectorizedArray holding packed
     * scalars).
     *
     * @ingroup ScalarConservationEquations
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

      using state_type = typename View::state_type;

      using flux_contribution_type = typename View::flux_contribution_type;

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
          , parameters_(limiter.parameters_.template view<MemorySpace>())
      {
      }

      /**
       * Return the number of limiter iterations.
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int iterations() const
      {
        return parameters_->iterations;
      }

      /**
       * Return the factor used for scaling the relaxation window.
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber relaxation_factor() const
      {
        return ScalarNumber(parameters_->relaxation_factor);
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
      DEAL_II_HOST_DEVICE Bounds combine_bounds(
          const Bounds &bounds_left, const Bounds &bounds_right) const;

      /**
       * This function applies a relaxation to a given a (strict) bound @p
       * bounds using a non dimensionalized measure @p hd (that should
       * scale as $h^d$, where $h$ is the local mesh size).
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
      const Limiter<ScalarNumber>::Parameters *const parameters_;

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


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    LimiterView<dim, Number, MemorySpace>::projection_bounds_from_state(
        const PrecomputedVectorView & /*pv*/,
        const unsigned int /*i*/,
        const state_type &U_i) const -> Bounds
    {
      const auto u_i = view_.state(U_i);
      return {/*u_min*/ u_i, /*u_max*/ u_i};
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    LimiterView<dim, Number, MemorySpace>::combine_bounds(
        const Bounds &bounds_left, const Bounds &bounds_right) const -> Bounds
    {
      const auto &[u_min_l, u_max_l] = bounds_left;
      const auto &[u_min_r, u_max_r] = bounds_right;

      return {std::min(u_min_l, u_min_r), std::max(u_max_l, u_max_r)};
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    LimiterView<dim, Number, MemorySpace>::fully_relax_bounds(
        const Bounds &bounds, const Number &hd) const -> Bounds
    {
      auto relaxed_bounds = bounds;
      auto &[u_min, u_max] = relaxed_bounds;

      /* Use r = factor * (m_i / |Omega|) ^ (1.5 / d): */

      Number r = std::sqrt(hd);                   // in 3D: ^ 3/6
      if constexpr (dim == 2)                     //
        r = ryujin::fixed_power<3>(std::sqrt(r)); // in 2D: ^ 3/4
      else if constexpr (dim == 1)                //
        r = ryujin::fixed_power<3>(r);            // in 1D: ^ 3/2
      r *= relaxation_factor();

      u_min = std::min((Number(1.) - r) * u_min, (Number(1.) + r) * u_min);
      u_max = std::max((Number(1.) + r) * u_max, (Number(1.) - r) * u_max);

      return relaxed_bounds;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    LimiterView<dim, Number, MemorySpace>::reset(
        const PrecomputedVectorView & /*pv*/,
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


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    LimiterView<dim, Number, MemorySpace>::accumulate(
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


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    LimiterView<dim, Number, MemorySpace>::bounds(const Number hd_i) const
        -> Bounds
    {
      const auto &[u_min, u_max] = bounds_;

      auto relaxed_bounds = fully_relax_bounds(bounds_, hd_i);
      auto &[u_min_relaxed, u_max_relaxed] = relaxed_bounds;

      /* Apply a stricter window: */

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();

      const Number u_relaxation =
          ScalarNumber(2. * relaxation_factor()) *
          std::abs(u_relaxation_numerator_) /
          (std::abs(u_relaxation_denominator_) + Number(eps));

      u_min_relaxed = std::max(u_min_relaxed, u_min - u_relaxation);
      u_max_relaxed = std::min(u_max_relaxed, u_max + u_relaxation);

      return relaxed_bounds;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE std::tuple<Number, bool>
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
      const ScalarNumber relax = ScalarNumber(1. + 10000. * eps);

      const auto &u_U = view_.state(U);
      const auto &u_P = view_.state(P);

      const auto &u_min = std::get<0>(bounds);
      const auto &u_max = std::get<1>(bounds);

      /*
       * Verify that u_U is within bounds. This property might be
       * violated for relative CFL numbers larger than 1.
       *
       * u_min, u_U, u_max might be negative, thus relax in both directions.
       */
      const auto test_max = std::max(
          Number(0.), std::min(u_U - relax * u_max, relax * u_U - u_max));
      const auto test_min = std::max(
          Number(0.), std::min(u_min - relax * u_U, relax * u_min - u_U));
      if (!(test_max == Number(0.) && test_min == Number(0.))) {
#ifdef DEBUG_OUTPUT
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Bounds violation: low-order state (critical)!"
                  << "\n\t\tu min:         " << u_min
                  << "\n\t\tu min (delta): " << negative_part(u_U - u_min)
                  << "\n\t\tu:             " << u_U
                  << "\n\t\tu max (delta): " << positive_part(u_U - u_max)
                  << "\n\t\tu max:         " << u_max << "\n"
                  << std::endl;
#endif
        success = false;
      }

      const auto regularization =
          Number(100. * std::numeric_limits<ScalarNumber>::min());

      const Number denominator =
          ScalarNumber(1.) /
          std::max(regularization, std::abs(u_P) + eps * u_max);

      t_r = ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
          u_max,
          u_U + t_r * u_P,
          /*
           * u_P is positive.
           *
           * Note: Do not take an absolute value here. If we are out of
           * bounds we have to ensure that t_r is set to t_min.
           */
          (u_max - u_U) * denominator,
          t_r);

      t_r = ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
          u_U + t_r * u_P,
          u_min,
          /*
           * u_P is negative.
           *
           * Note: Do not take an absolute value here. If we are out of
           * bounds we have to ensure that t_r is set to t_min.
           */
          (u_U - u_min) * denominator,
          t_r);

      /*
       * Ensure that t_min <= t <= t_max. This might not be the case if
       * u_U is outside the interval [u_min, u_max]. Furthermore,
       * the quotient we take above is prone to numerical cancellation in
       * particular in the second pass of the limiter when u_P might be
       * small.
       */
      t_r = std::min(t_r, t_max);
      t_r = std::max(t_r, t_min);

#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
      /*
       * Verify that the new state is within bounds:
       *
       * u_min, u_U, u_max might be negative, thus relax in both directions.
       */
      const auto u_new = view_.state(U + t_r * P);
      const auto test_new_max = std::max(
          Number(0.), std::min(u_new - relax * u_max, relax * u_new - u_max));
      const auto test_new_min = std::max(
          Number(0.), std::min(u_min - relax * u_new, relax * u_min - u_new));
      if (!(test_new_max == Number(0.) && test_new_min == Number(0.))) {
#ifdef DEBUG_OUTPUT
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Bounds violation: high-order state!"
                  << "\n\t\tu min:         " << u_min
                  << "\n\t\tu min (delta): " << negative_part(u_new - u_min)
                  << "\n\t\tu:             " << u_new
                  << "\n\t\tu max (delta): " << positive_part(u_new - u_max)
                  << "\n\t\tu max:         " << u_max << "\n"
                  << std::endl;
#endif
        success = false;
      }
#endif

      return {t_r, success};
    }
  } // namespace ScalarConservation
} // namespace ryujin
