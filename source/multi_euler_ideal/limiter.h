//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2025 by the ryujin authors
// Copyright (C) 2025 by Triad National Security, LLC
//

#pragma once

#include "hyperbolic_system.h"

#include <compile_time_options.h>
#include <multicomponent_vector.h>
#include <newton.h>
#include <simd.h>

namespace ryujin
{
  namespace MultiSpeciesEuler
  {
    template <typename ScalarNumber = double>
    class LimiterParameters : public dealii::ParameterAcceptor
    {
    public:
      LimiterParameters(const std::string &subsection = "/Limiter")
          : ParameterAcceptor(subsection)
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

        relaxation_factor_ = ScalarNumber(1.);
        add_parameter("relaxation factor",
                      relaxation_factor_,
                      "Factor for scaling the relaxation window with r_i = "
                      "factor * (m_i/|Omega|)^(1.5/d).");
      }

      ACCESSOR_READ_ONLY(iterations);
      ACCESSOR_READ_ONLY(newton_tolerance);
      ACCESSOR_READ_ONLY(relaxation_factor);

    private:
      unsigned int iterations_;
      ScalarNumber newton_tolerance_;
      ScalarNumber relaxation_factor_;
    };


    /**
     * The convex limiter for the multi-species Euler equations.
     *
     * The class implements a convex limiting technique as described in
     * @cite GuermondEtAl2018, @cite ryujin-2021-1, @cite ryujin-2023-4,
     * and extended to multi-species flows in [ClaytonDzanicTovar-2025].
     *
     * For the multi-species case, we enforce bounds on (see Section 5 in
     * [ClaytonDzanicTovar-2025]):
     *   - Partial densities: \f$(\alpha_k \rho_k)_{\min} \le \alpha_k
     *     \rho_k \le (\alpha_k \rho_k)_{\max}\f$ for each species \f$k\f$
     *   - Internal energy: \f$\varepsilon \ge \varepsilon_{\min}\f$
     *   - Specific entropy: \f$s(\mathbf{u}) \ge s_{\min}\f$
     *
     * Given a computed set of bounds and an update direction
     * \f$\mathbf{P}_{ij}\f$ one determines a candidate \f$\tilde l_{ij}\f$
     * by computing:
     * \f{align}
     *   \tilde l_{ij} = \max_{l\,\in\,[0,1]}
     *   \,\Big\{(\alpha_k \rho_k)_{\min}\,\le\,\alpha_k \rho_k\,(\mathbf{U}_i
     *   + \tilde l_{ij}\mathbf{P}_{ij}) \,\le\,(\alpha_k \rho_k)_{\max},\;
     *   s_{\min}\,\le\,s\,(\mathbf{U}_{i}+\tilde l_{ij}\mathbf{P}_{ij})\Big\}.
     * \f}
     *
     * Algorithmically this is accomplished as follows: Given an initial
     * interval \f$[t_L,t_R]\f$, where \f$t_L\f$ is a good state, we first
     * make the interval smaller ensuring the bounds on the partial
     * densities are fulfilled. We then perform a quadratic Newton
     * iteration (updating \f$[t_L,t_R]\f$) solving for the root of the
     * 3-convex function:
     * \f{align}
     *   \Psi(\mathbf{U})\;=\;\rho^{\bar{\gamma}+1}(\mathbf{U})\,
     *   \big(s(\mathbf{U})-s_{\min}\big),
     * \f}
     * where we use a surrogate gamma \f$\bar{\gamma} = \gamma_{\min}\f$
     * to ensure convexity of \f$\Psi\f$.
     *
     * @ingroup MultiSpeciesEulerEquations
     */
    template <int dim, typename Number = double>
    class Limiter
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

      using PrecomputedVector = typename View::PrecomputedVector;

      using Parameters = LimiterParameters<ScalarNumber>;

      //@}
      /**
       * @name Computation and manipulation of bounds
       */
      //
      //@{
      /**
       * The number of stored entries in the bounds array:
       * 2 * n_species (min/max per species) + 1 (rho_e_min) + 1 (s_min)
       */
      static constexpr unsigned int n_bounds = 2 * n_species + 2;

      /**
       * Array type used to store accumulated bounds.
       */
      using Bounds = std::array<Number, n_bounds>;

      /**
       * Constructor taking a HyperbolicSystem instance as argument
       */
      Limiter(const HyperbolicSystem &hyperbolic_system,
              const Parameters &parameters,
              const PrecomputedVector &precomputed_values)
          : hyperbolic_system(hyperbolic_system)
          , parameters(parameters)
          , precomputed_values(precomputed_values)
      {
      }

      /**
       * Given a state @p U_i and an index @p i return "strict" bounds,
       * i.e., a minimal convex set containing the state.
       */
      Bounds projection_bounds_from_state(const unsigned int i,
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
       * Limiter<dim, Number> limiter;
       * for (unsigned int i = n_internal; i < n_owned; ++i) {
       *   // ...
       *   limiter.reset(i, U_i, flux_i);
       *   for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
       *     // ...
       *     limiter.accumulate(js, U_j, flux_j, scaled_c_ij, affine_shift);
       *   }
       *   limiter.bounds(hd_i);
       * }
       * ```
       */
      //@{

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
      void accumulate(const unsigned int *js,
                      const state_type &U_j,
                      const flux_contribution_type &flux_j,
                      const dealii::Tensor<1, dim, Number> &scaled_c_ij,
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
      /** @name Arguments and internal fields */
      //@{

      const HyperbolicSystem &hyperbolic_system;
      const Parameters &parameters;
      const PrecomputedVector &precomputed_values;

      state_type U_i;
      flux_contribution_type flux_i;

      Bounds bounds_;

      dealii::Tensor<1, n_species, Number> rho_relaxation_numerator;
      Number rho_relaxation_denominator;
      Number rho_e_interp_max;
      Number s_interp_max;

      Number cv_i;

      //@}
    };


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    Limiter<dim, Number>::projection_bounds_from_state(
        const unsigned int i, const state_type &U_i) const -> Bounds
    {
      const auto view = hyperbolic_system.view<dim, Number>();

      const auto &[rho_i, p_i, gamma_min_i, s_i, harten_i] =
          precomputed_values.template get_tensor<Number, precomputed_type>(i);

      Bounds result;
      /* Partial density bounds: [rho_k_min, rho_k_max] for each species */
      for (unsigned int k = 0; k < n_species; ++k) {
        const auto alpha_rho_k = view.partial_density(U_i, k);
        result[2 * k] = alpha_rho_k;     /* min */
        result[2 * k + 1] = alpha_rho_k; /* max */
      }
      /* Internal energy and entropy bounds */
      result[2 * n_species] = view.internal_energy(U_i); /* rho_e_min */
      result[2 * n_species + 1] = s_i;                   /* s_min */

      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto Limiter<dim, Number>::combine_bounds(
        const Bounds &bounds_left, const Bounds &bounds_right) const -> Bounds
    {
      Bounds result;
      /* Combine partial density bounds for each species */
      for (unsigned int k = 0; k < n_species; ++k) {
        result[2 * k] = std::min(bounds_left[2 * k], bounds_right[2 * k]);
        result[2 * k + 1] =
            std::max(bounds_left[2 * k + 1], bounds_right[2 * k + 1]);
      }
      /* Combine internal energy and entropy bounds */
      result[2 * n_species] =
          std::min(bounds_left[2 * n_species], bounds_right[2 * n_species]);
      result[2 * n_species + 1] = std::min(bounds_left[2 * n_species + 1],
                                           bounds_right[2 * n_species + 1]);

      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    Limiter<dim, Number>::fully_relax_bounds(const Bounds &bounds,
                                             const Number &hd) const -> Bounds
    {
      auto relaxed_bounds = bounds;

      /* Use r = factor * (m_i / |Omega|) ^ (1.5 / d): */

      Number r = std::sqrt(hd);                              // in 3D: ^ 3/6
      if constexpr (dim == 2)                                //
        r = dealii::Utilities::fixed_power<3>(std::sqrt(r)); // in 2D: ^ 3/4
      else if constexpr (dim == 1)                           //
        r = dealii::Utilities::fixed_power<3>(r);            // in 1D: ^ 3/2
      r *= parameters.relaxation_factor();

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();

      /* Relax partial density bounds for each species */
      for (unsigned int k = 0; k < n_species; ++k) {
        relaxed_bounds[2 * k] *= std::max(Number(1.) - r, Number(eps));
        relaxed_bounds[2 * k + 1] *= (Number(1.) + r);
      }

      /* Relax internal energy bound */
      relaxed_bounds[2 * n_species] *= std::max(Number(1.) - r, Number(eps));

      /* Relax entropy bound */
      auto &s_min_relaxed = relaxed_bounds[2 * n_species + 1];
      const Number scaled_ent = std::exp(s_min_relaxed / cv_i);
      s_min_relaxed = cv_i * std::log((Number(1.) - r) * scaled_ent);

      return relaxed_bounds;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void
    Limiter<dim, Number>::reset(const unsigned int /* i */,
                                const state_type &new_U_i,
                                const flux_contribution_type &new_flux_i)
    {
      const auto view = hyperbolic_system.view<dim, Number>();

      U_i = new_U_i;
      flux_i = new_flux_i;

      cv_i = view.cv_mixture(new_U_i);

      /* Initialize bounds for each species */
      for (unsigned int k = 0; k < n_species; ++k) {
        bounds_[2 * k] = Number(std::numeric_limits<ScalarNumber>::max());
        bounds_[2 * k + 1] = Number(0.);
      }
      bounds_[2 * n_species] = Number(std::numeric_limits<ScalarNumber>::max());
      bounds_[2 * n_species + 1] =
          Number(std::numeric_limits<ScalarNumber>::max());

      /* Relaxation: */
      for (unsigned int k = 0; k < n_species; ++k)
        rho_relaxation_numerator[k] = Number(0.);
      rho_relaxation_denominator = Number(0.);

      rho_e_interp_max = Number(0.);
      s_interp_max = Number(0.);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void Limiter<dim, Number>::accumulate(
        const unsigned int *js,
        const state_type &U_j,
        const flux_contribution_type &flux_j,
        const dealii::Tensor<1, dim, Number> &scaled_c_ij,
        const state_type &affine_shift)
    {
      // TODO: Currently we only apply the affine_shift to U_ij_bar (which
      // then enters all bounds), but we do not modify s_interp and
      // rho_relaxation. When actually adding a source term to the Euler
      // equations verify that this does the right thing.
      Assert(std::max(affine_shift.norm(), Number(0.)) == Number(0.),
             dealii::ExcNotImplemented());

      const auto view = hyperbolic_system.view<dim, Number>();

      const auto rho_e_j = view.internal_energy(U_j);

      const auto [rho_j, p_j, gamma_min_j, s_j, harten_j] =
          precomputed_values.template get_tensor<Number, precomputed_type>(js);

      /* bar state shifted by an affine shift: */
      const auto U_ij_bar =
          ScalarNumber(0.5) * (U_i + U_j) -
          ScalarNumber(0.5) * contract(add(flux_j, -flux_i), scaled_c_ij) +
          affine_shift;

      const auto rho_e_ij_bar = view.internal_energy(U_ij_bar);

      /* Density bounds and relaxation for each species: */
      const auto beta_ij = Number(1.);
      for (unsigned int k = 0; k < n_species; ++k) {
        const auto alpha_rho_k_i = view.partial_density(U_i, k);
        const auto alpha_rho_k_j = view.partial_density(U_j, k);
        const auto alpha_rho_k_ij_bar = view.partial_density(U_ij_bar, k);

        bounds_[2 * k] = std::min(bounds_[2 * k], alpha_rho_k_ij_bar);
        bounds_[2 * k + 1] = std::max(bounds_[2 * k + 1], alpha_rho_k_ij_bar);

        rho_relaxation_numerator[k] +=
            beta_ij * (alpha_rho_k_i + alpha_rho_k_j);
      }
      rho_relaxation_denominator += std::abs(beta_ij);

      /* Internal energy bounds and relaxation */
      auto &rho_e_min = bounds_[2 * n_species];
      rho_e_min = std::min(rho_e_min, rho_e_j);
      rho_e_min = std::min(rho_e_min, rho_e_ij_bar);

      const auto rho_e_interp =
          view.internal_energy((U_i + U_j) * ScalarNumber(.5));
      rho_e_interp_max = std::max(rho_e_interp_max, rho_e_interp);

      /* Entropy bounds and relaxation: */
      auto &s_min = bounds_[2 * n_species + 1];
      const auto s_ij_bar = view.specific_entropy(U_ij_bar);

      const Number s_interp =
          view.specific_entropy((U_i + U_j) * ScalarNumber(.5));

      s_min = std::min(s_min, s_j);
      s_min = std::min(s_min, s_ij_bar);
      s_interp_max = std::max(s_interp_max, s_interp);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    Limiter<dim, Number>::bounds(const Number hd_i) const -> Bounds
    {
      auto relaxed_bounds = fully_relax_bounds(bounds_, hd_i);

      /* Apply a stricter window: */
      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();

      /* Apply stricter bounds for each species */
      for (unsigned int k = 0; k < n_species; ++k) {
        const auto rho_k_relaxation =
            ScalarNumber(2. * parameters.relaxation_factor()) *
            std::abs(rho_relaxation_numerator[k]) /
            (std::abs(rho_relaxation_denominator) + Number(eps));

        relaxed_bounds[2 * k] =
            std::max(relaxed_bounds[2 * k], bounds_[2 * k] - rho_k_relaxation);
        relaxed_bounds[2 * k + 1] = std::min(
            relaxed_bounds[2 * k + 1], bounds_[2 * k + 1] + rho_k_relaxation);
      }

      /* Apply stricter bounds for internal energy and entropy */
      const auto &rho_e_min = bounds_[2 * n_species];
      const auto &s_min = bounds_[2 * n_species + 1];

      const auto int_relaxation =
          parameters.relaxation_factor() * (rho_e_interp_max - rho_e_min);
      const auto entropy_relaxation =
          parameters.relaxation_factor() * (s_interp_max - s_min);

      relaxed_bounds[2 * n_species] =
          std::max(relaxed_bounds[2 * n_species], rho_e_min - int_relaxation);
      relaxed_bounds[2 * n_species + 1] = std::max(
          relaxed_bounds[2 * n_species + 1], s_min - entropy_relaxation);

      return relaxed_bounds;
    }
  } // namespace MultiSpeciesEuler
} // namespace ryujin
