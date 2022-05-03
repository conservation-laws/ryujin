//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include <hyperbolic_system.h>

#include <compile_time_options.h>
#include <multicomponent_vector.h>
#include <newton.h>
#include <simd.h>

namespace ryujin
{
  template <int dim, typename Number = double>
  class Limiter
  {
  public:
    /**
     * @copydoc HyperbolicSystem::problem_dimension
     */
    // clang-format off
    static constexpr unsigned int problem_dimension = HyperbolicSystem::problem_dimension<dim>;
    // clang-format on

    /**
     * @copydoc HyperbolicSystem::state_type
     */
    using state_type = HyperbolicSystem::state_type<dim, Number>;

    /**
     * @copydoc HyperbolicSystem::ScalarNumber
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
     *   limiter.reset(i, U_i);
     *   for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
     *     // ...
     *     limiter.accumulate(js, U_j, scaled_c_ij, beta_ij);
     *   }
     *   limiter.apply_relaxation(hd_i);
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
     * The number of precomputed values.
     */
    static constexpr unsigned int n_precomputed_values = 1;

    /**
     * Array type used for precomputed values.
     */
    using PrecomputedValues = std::array<Number, n_precomputed_values>;

    /**
     * Precomputed values for a given state.
     */
    static PrecomputedValues
    precompute_values(const HyperbolicSystem &hyperbolic_system,
                      const state_type &U);

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
                    const dealii::Tensor<1, dim, Number> &scaled_c_ij,
                    const Number beta_ij);

    /**
     * Apply relaxation.
     */
    void apply_relaxation(const Number hd_i);

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

    //@}
  };


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline typename Limiter<dim, Number>::PrecomputedValues
  Limiter<dim, Number>::precompute_values(
      const HyperbolicSystem &hyperbolic_system, const state_type &U_i)
  {
    PrecomputedValues result;
    return result;
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void Limiter<dim, Number>::reset(unsigned int i)
  {
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void Limiter<dim, Number>::accumulate(
      const unsigned int *js,
      const state_type &U_i,
      const state_type &U_j,
      const dealii::Tensor<1, dim, Number> &scaled_c_ij,
      const Number beta_ij)
  {
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void
  Limiter<dim, Number>::apply_relaxation(Number hd_i)
  {
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
