//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// [LANL Copyright Statement]
// Copyright (C) 2020 - 2026 by the ryujin authors
// Copyright (C) 2025 - 2026 by Triad National Security, LLC
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/gpu.h>
#include <ryujin/base/observer_pointer.h>
#include <ryujin/base/simd.h>
#include <ryujin/euler_atmospheric/hyperbolic_system.h>
#include <ryujin/linear_algebra/multicomponent_vector.h>

#include <deal.II/base/config.h>

namespace ryujin
{
  namespace EulerAtmospheric
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
     *   \theta_{\text{min}}\,\le\,\theta\,(\mathbf U_{i}+\tilde l_{ij}\mathbf
     * P_{ij})\,\le\,\theta_{\text{max}}\Big\}, \f}
     *
     * where \f$\theta\f$ denotes the potential temperature.
     *
     * Algorithmically this is accomplished as follows: Given an initial
     * interval \f$[t_L,t_R]\f$, where \f$t_L\f$ is a good state, we first
     * make the interval smaller ensuring the bounds on the density are
     * fulfilled. We then make the interval smaller again by enforcing the
     * two (linear in \f$t\f$) constraints
     * \f{align}
     *   (\rho\theta) - \theta_{\text{min}}\,\rho \ge 0,
     *   \qquad
     *   \theta_{\text{max}}\,\rho - (\rho\theta) \ge 0.
     * \f}
     *
     * @ingroup EulerAtmosphericEquations
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
        double newton_tolerance;
        unsigned int newton_max_iterations;
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
          , parameters_("euler_atmospheric_limiter_parameters",
                        TransferPolicy::implicit_transfers_host_resident)
          , hyperbolic_system_(&hyperbolic_system)
      {
        /* reference remains valid due to implicit_transfers_host_resident */
        auto &parameters = *parameters_.view();

        parameters.iterations = 2;
        add_parameter("iterations",
                      parameters.iterations,
                      "Number of limiter iterations");

        if constexpr (std::is_same<ScalarNumber, double>::value)
          parameters.newton_tolerance = 1.e-10;
        else
          parameters.newton_tolerance = 1.e-4;
        add_parameter("newton tolerance",
                      parameters.newton_tolerance,
                      "Tolerance for the quadratic newton stopping criterion");

        parameters.newton_max_iterations = 2;
        add_parameter("newton max iterations",
                      parameters.newton_max_iterations,
                      "Maximal number of quadratic newton iterations performed "
                      "during limiting");

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
     * @ingroup EulerAtmosphericEquations
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
      static constexpr unsigned int n_bounds = 4;

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
       * scale as $h^d$, where $h$ is the local mesh size). This is done
       * for the case of the atmospheric Euler equations by multiplying
       * maximum bounds with $(1+r)$ and minimum bounds with $(1-r)$, while
       * ensuring that the bounds still describe an admissible state.
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
       *                             affine_shift);
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
      const Limiter<ScalarNumber>::Parameters *const parameters_;

      flux_contribution_type flux_i_;

      Bounds bounds_;

      Number rho_relaxation_numerator_;
      Number rho_relaxation_denominator_;
      Number theta_relaxation_numerator_;
      Number theta_relaxation_denominator_;

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
      const auto rho_i = view_.density(U_i);
      const auto theta_i = view_.potential_temperature(U_i) / rho_i;

      return {/*rho_min*/ rho_i,
              /*rho_max*/ rho_i,
              /*theta_min*/ theta_i,
              /*theta_max*/ theta_i};
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    LimiterView<dim, Number, MemorySpace>::combine_bounds(
        const Bounds &bounds_left, const Bounds &bounds_right) const -> Bounds
    {
      const auto &[rho_min_l, rho_max_l, theta_min_l, theta_max_l] =
          bounds_left;
      const auto &[rho_min_r, rho_max_r, theta_min_r, theta_max_r] =
          bounds_right;

      return {std::min(rho_min_l, rho_min_r),
              std::max(rho_max_l, rho_max_r),
              std::min(theta_min_l, theta_min_r),
              std::max(theta_max_l, theta_max_r)};
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    LimiterView<dim, Number, MemorySpace>::fully_relax_bounds(
        const Bounds &bounds, const Number &hd) const -> Bounds
    {
      auto relaxed_bounds = bounds;
      auto &[rho_min, rho_max, theta_min, theta_max] = relaxed_bounds;

      /* Use r = factor * (m_i / |Omega|) ^ (1.5 / d): */

      Number r = std::sqrt(hd);                   // in 3D: ^ 3/6
      if constexpr (dim == 2)                     //
        r = ryujin::fixed_power<3>(std::sqrt(r)); // in 2D: ^ 3/4
      else if constexpr (dim == 1)                //
        r = ryujin::fixed_power<3>(r);            // in 1D: ^ 3/2
      r *= relaxation_factor();

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
      rho_min *= std::max(Number(1.) - r, Number(eps));
      rho_max *= (Number(1.) + r);
      theta_min *= std::max(Number(1.) - r, Number(eps));
      theta_max *= (Number(1.) + r);

      return relaxed_bounds;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    LimiterView<dim, Number, MemorySpace>::reset(
        const PrecomputedVectorView & /*pv*/,
        const unsigned int /*i*/,
        const state_type & /*U_i*/,
        const flux_contribution_type &flux_i)
    {
      flux_i_ = flux_i;

      /* Bounds: */

      auto &[rho_min, rho_max, theta_min, theta_max] = bounds_;

      rho_min = Number(std::numeric_limits<ScalarNumber>::max());
      rho_max = Number(0.);
      theta_min = Number(std::numeric_limits<ScalarNumber>::max());
      theta_max = Number(0.);

      /* Relaxation: */

      rho_relaxation_numerator_ = Number(0.);
      rho_relaxation_denominator_ = Number(0.);
      theta_relaxation_numerator_ = Number(0.);
      theta_relaxation_denominator_ = Number(0.);
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
      const auto &U_i = std::get<0>(flux_i_);
      const auto &[U_star_ij, U_star_ji] =
          view_.equilibrated_states(flux_i_, flux_j);

      /*
       * The bar states. We only need the density and the potential
       * temperature component, so we can use the flux g() that omits the
       * pressure:
       */

      const auto g_star_ij = view_.g(U_star_ij);
      const auto g_star_ji = view_.g(U_star_ji);

      /* bar state shifted by an affine shift: */
      const auto U_ij_bar =
          ScalarNumber(0.5) *
              (U_star_ij + U_star_ji +
               contract(add(g_star_ij, -g_star_ji), scaled_c_ij)) +
          affine_shift;

      /* Bounds: */

      auto &[rho_min, rho_max, theta_min, theta_max] = bounds_;

      const auto rho_ij_bar = view_.density(U_ij_bar);
      const auto theta_ij_bar =
          view_.potential_temperature(U_ij_bar) / rho_ij_bar;

      rho_min = std::min(rho_min, rho_ij_bar);
      rho_max = std::max(rho_max, rho_ij_bar);
      theta_min = std::min(theta_min, theta_ij_bar);
      theta_max = std::max(theta_max, theta_ij_bar);

      /* Relaxation: */

      const auto rho_i = view_.density(U_i);
      const auto rho_j = view_.density(U_j);
      const auto theta_i = view_.potential_temperature(U_i) / rho_i;
      const auto theta_j = view_.potential_temperature(U_j) / rho_j;

      /* Use a uniform weight. */
      const auto beta_ij = Number(1.);
      rho_relaxation_numerator_ += beta_ij * (rho_i + rho_j);
      rho_relaxation_denominator_ += std::abs(beta_ij);
      theta_relaxation_numerator_ += beta_ij * (theta_i + theta_j);
      theta_relaxation_denominator_ += std::abs(beta_ij);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    LimiterView<dim, Number, MemorySpace>::bounds(const Number hd_i) const
        -> Bounds
    {
      const auto &[rho_min, rho_max, theta_min, theta_max] = bounds_;

      auto relaxed_bounds = fully_relax_bounds(bounds_, hd_i);
      auto &[rho_min_relaxed,
             rho_max_relaxed,
             theta_min_relaxed,
             theta_max_relaxed] = relaxed_bounds;

      /* Apply a stricter window: */

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();

      const auto rho_relaxation =
          ScalarNumber(2. * relaxation_factor()) *
          std::abs(rho_relaxation_numerator_) /
          (std::abs(rho_relaxation_denominator_) + Number(eps));

      const auto theta_relaxation =
          ScalarNumber(2. * relaxation_factor()) *
          std::abs(theta_relaxation_numerator_) /
          (std::abs(theta_relaxation_denominator_) + Number(eps));

      rho_min_relaxed = std::max(rho_min_relaxed, rho_min - rho_relaxation);
      rho_max_relaxed = std::min(rho_max_relaxed, rho_max + rho_relaxation);
      theta_min_relaxed =
          std::max(theta_min_relaxed, theta_min - theta_relaxation);
      theta_max_relaxed =
          std::min(theta_max_relaxed, theta_max + theta_relaxation);

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

#ifdef NO_LIMITING
      return {t_r, success};
#endif

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
      const auto large = view_.vacuum_state_relaxation_large();
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
       * Then limit the potential temperature theta = (rho theta) / rho by
       * enforcing the two (linear in t) constraints
       *
       *   Psi_3(U) = (rho theta) - theta_min * rho >= 0,
       *   Psi_4(U) = theta_max * rho - (rho theta) >= 0.
       */

      {
        const auto &rho_U = view_.density(U);
        const auto &rho_P = view_.density(P);
        const auto &rho_theta_U = view_.potential_temperature(U);
        const auto &rho_theta_P = view_.potential_temperature(P);

        const auto &theta_min = std::get<2>(bounds);
        const auto &theta_max = std::get<3>(bounds);

        /*
         * Verify that theta_U is within bounds. This property might be
         * violated for relative CFL numbers larger than 1.
         */
        const auto test_min =
            std::max(Number(0.), rho_theta_U - relax * theta_max * rho_U);
        const auto test_max =
            std::max(Number(0.), theta_min * rho_U - relax * rho_theta_U);
        if (!(test_min == Number(0.) && test_max == Number(0.))) {
#ifdef DEBUG_OUTPUT
          std::cout << std::fixed << std::setprecision(16);
          std::cout
              << "Bounds violation: low-order potential temperature (critical)!"
              << "\n\t\ttheta min:         " << theta_min
              << "\n\t\ttheta min (delta): "
              << negative_part(rho_theta_U / rho_U - theta_min)
              << "\n\t\ttheta:             " << rho_theta_U / rho_U
              << "\n\t\ttheta max (delta): "
              << positive_part(rho_theta_U / rho_U - theta_max)
              << "\n\t\ttheta max:         " << theta_max << "\n"
              << std::endl;
#endif
          success = false;
        }

        const auto psi_min_U = rho_theta_U - theta_min * rho_U;
        const auto psi_min_P = rho_theta_P - theta_min * rho_P;
        const auto psi_max_U = theta_max * rho_U - rho_theta_U;
        const auto psi_max_P = theta_max * rho_P - rho_theta_P;

        const Number denominator_scale = eps * theta_max * std::abs(rho_U);

        constexpr auto lt = dealii::SIMDComparison::less_than;

        Number t_q = t_r;

        t_q = ryujin::compare_and_apply_mask<lt>( //
            psi_min_U + t_q * psi_min_P,
            Number(0.),
            /*
             * psi_min_P is negative.
             *
             * Note: Do not take an absolute value of psi_min_U here. If we
             * are out of bounds we have to ensure that t_q is set to t_min.
             */
            psi_min_U / (std::abs(psi_min_P) + denominator_scale),
            t_q);

        t_q = ryujin::compare_and_apply_mask<lt>( //
            psi_max_U + t_q * psi_max_P,
            Number(0.),
            /*
             * psi_max_P is negative.
             *
             * Note: Do not take an absolute value of psi_max_U here. If we
             * are out of bounds we have to ensure that t_q is set to t_min.
             */
            psi_max_U / (std::abs(psi_max_P) + denominator_scale),
            t_q);

        /*
         * Ensure that t_min <= t <= t_max. This might not be the case if
         * theta_U is outside the interval [theta_min, theta_max].
         * Furthermore, the quotient we take above is prone to numerical
         * cancellation in particular in the second pass of the limiter when
         * P might be small.
         */
        t_r = std::min(t_q, t_r);
        t_r = std::min(t_r, t_max);
        t_r = std::max(t_r, t_min);

#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
        /*
         * Verify that the new state is within bounds:
         */
        const auto U_new = U + t_r * P;
        const auto rho_new = view_.density(U_new);
        const auto rho_theta_new = view_.potential_temperature(U_new);
        const auto test_new_min =
            std::max(Number(0.), rho_theta_new - relax * theta_max * rho_new);
        const auto test_new_max =
            std::max(Number(0.), theta_min * rho_new - relax * rho_theta_new);
        if (!(test_new_min == Number(0.) && test_new_max == Number(0.))) {
#ifdef DEBUG_OUTPUT
          std::cout << std::fixed << std::setprecision(16);
          std::cout << "Bounds violation: high-order potential temperature!"
                    << "\n\t\ttheta min:         " << theta_min
                    << "\n\t\ttheta min (delta): "
                    << negative_part(rho_theta_new / rho_new - theta_min)
                    << "\n\t\ttheta:             " << rho_theta_new / rho_new
                    << "\n\t\ttheta max (delta): "
                    << positive_part(rho_theta_new / rho_new - theta_max)
                    << "\n\t\ttheta max:         " << theta_max << "\n"
                    << std::endl;
#endif
          success = false;
        }
#endif
      }

      return {t_r, success};
    }
  } // namespace EulerAtmospheric
} // namespace ryujin
