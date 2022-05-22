//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <hyperbolic_system.h>

#include <compile_time_options.h>
#include <multicomponent_vector.h>
#include <newton.h>
#include <simd.h>

namespace ryujin
{
  /**
   * The convex limiter.
   *
   * The class implements a convex limiting technique as described in
   * @cite GuermondEtAl2018 and @cite ryujin-2022-1. Given a
   * computed set of bounds and an update direction \f$\mathbf P_{ij}\f$
   * one can now determine a candidate \f$\tilde l_{ij}\f$ by computing
   *
   * \f{align}
   *   \tilde l_{ij} = \max_{l\,\in\,[0,1]}
   *   \,\Big\{\rho_{\text{min}}\,\le\,\rho\,(\mathbf U_i +\tilde l_{ij}\mathbf
   * P_{ij})
   *   \,\le\,\rho_{\text{max}},\quad
   *   \phi_{\text{min}}\,\le\,\phi\,(\mathbf U_{i}+\tilde l_{ij}\mathbf
   * P_{ij})\Big\}, \f}
   *
   * where \f$\psi\f$ denots the specific entropy @cite ryujin-2022-1.
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
     * @copydoc HyperbolicSystem::prec_type
     */
    using prec_type = HyperbolicSystem::prec_type<dim, Number>;

    /**
     * @copydoc HyperbolicSystem::ScalarNumber
     */
    using ScalarNumber = typename get_value_type<Number>::type;

    /**
     * @name Precomputation of indicator quantities
     */
    //@{

    /**
     * The number of precomputed values.
     */
    static constexpr unsigned int n_precomputed_values = 1;

    /**
     * Array type used for precomputed values.
     */
    using precomputed_type = std::array<Number, n_precomputed_values>;

    /**
     * Precomputed values for a given state.
     */
    static precomputed_type
    precompute_values(const HyperbolicSystem &hyperbolic_system,
                      const state_type &U);

    //@}
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
     *     limiter.accumulate(js, U_i, U_j, pre_i, pre_j, scaled_c_ij, beta_ij);
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
                    const prec_type &prec_i,
                    const prec_type &prec_j,
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

    Number rho_relaxation_numerator;
    Number rho_relaxation_denominator;
    Number s_interp_max;

    //@}
  };


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline typename Limiter<dim, Number>::precomputed_type
  Limiter<dim, Number>::precompute_values(
      const HyperbolicSystem &hyperbolic_system, const state_type &U_i)
  {
    precomputed_type result;
    result[0] = hyperbolic_system.specific_entropy(U_i);
    return result;
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void Limiter<dim, Number>::reset(unsigned int i)
  {
    /* Bounds: */

    auto &[rho_min, rho_max, s_min] = bounds_;

    rho_min = Number(std::numeric_limits<ScalarNumber>::max());
    rho_max = Number(0.);

    const auto &[specific_entropy] =
        precomputed_values.template get_tensor<Number, precomputed_type>(i);

    s_min = specific_entropy;

    /* Relaxation: */

    rho_relaxation_numerator = Number(0.);
    rho_relaxation_denominator = Number(0.);
    s_interp_max = Number(0.);
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void Limiter<dim, Number>::accumulate(
      const unsigned int *js,
      const state_type &U_i,
      const state_type &U_j,
      const prec_type & /*prec_i*/,
      const prec_type & /*prec_j*/,
      const dealii::Tensor<1, dim, Number> &scaled_c_ij,
      const Number beta_ij)
  {
    /* Bounds: */

    auto &[rho_min, rho_max, s_min] = bounds_;

    const auto rho_i = hyperbolic_system.density(U_i);
    const auto m_i = hyperbolic_system.momentum(U_i);
    const auto rho_j = hyperbolic_system.density(U_j);
    const auto m_j = hyperbolic_system.momentum(U_j);
    const auto rho_ij_bar =
        ScalarNumber(0.5) * (rho_i + rho_j + (m_i - m_j) * scaled_c_ij);
    rho_min = std::min(rho_min, rho_ij_bar);
    rho_max = std::max(rho_max, rho_ij_bar);

    const auto &[specific_entropy_j] =
        precomputed_values.template get_tensor<Number, precomputed_type>(js);
    s_min = std::min(s_min, specific_entropy_j);

    /* Relaxation: */

    rho_relaxation_numerator += beta_ij * (rho_i + rho_j);
    rho_relaxation_denominator += beta_ij;

    const Number s_interp =
        hyperbolic_system.specific_entropy((U_i + U_j) * ScalarNumber(.5));
    s_interp_max = std::max(s_interp_max, s_interp);
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void
  Limiter<dim, Number>::apply_relaxation(Number hd_i, ScalarNumber factor)
  {
    auto &[rho_min, rho_max, s_min] = bounds_;

    /* Use r_i = factor * (m_i / |Omega|) ^ (1.5 / d): */

    Number r_i = std::sqrt(hd_i);                              // in 3D: ^ 3/6
    if constexpr (dim == 2)                                    //
      r_i = dealii::Utilities::fixed_power<3>(std::sqrt(r_i)); // in 2D: ^ 3/4
    else if constexpr (dim == 1)                               //
      r_i = dealii::Utilities::fixed_power<3>(r_i);            // in 1D: ^ 3/2
    r_i *= factor;

    constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
    const Number rho_relaxation =
        std::abs(rho_relaxation_numerator) /
        (std::abs(rho_relaxation_denominator) + Number(eps));

    rho_min = std::max((Number(1.) - Number(5.) * r_i) * rho_min,
                       rho_min - rho_relaxation);
    rho_max = std::min((Number(1.) + Number(5.) * r_i) * rho_max,
                       rho_max + rho_relaxation);

    s_min =
        std::max((Number(1.) - r_i) * s_min, Number(2.) * s_min - s_interp_max);
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

} // namespace ryujin
