//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "newton.h"
#include "offline_data.h"
#include "simd.h"

#include "problem_description.h"

namespace ryujin
{
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
   *   \,\Big\{\rho_{\text{min}}\,\le\,\rho\,(\mathbf U_i +\tilde l_{ij}\mathbf
   * P_{ij})
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
   * @ingroup EulerModule
   */
  template <int dim, typename Number = double>
  class Limiter
  {
  public:
    /**
     * @copydoc ProblemDescription::problem_dimension
     */
    // clang-format off
    static constexpr unsigned int problem_dimension = ProblemDescription::problem_dimension<dim>;
    // clang-format on

    /**
     * @copydoc ProblemDescription::state_type
     */
    using state_type = ProblemDescription::state_type<dim, Number>;

    /**
     * @copydoc ProblemDescription::ScalarNumber
     */
    using ScalarNumber = typename get_value_type<Number>::type;

    //@}
    /**
     * @name Stencil-based accumulations of bounds
     *
     * Intended usage:
     * ```
     * Limiter<dim, Number> limiter;
     * for (unsigned int i = n_internal; i < n_owned; ++i) {
     *   // ...
     *   limiter.reset(specific_entropy_i);
     *   for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
     *     // ...
     *     limiter.accumulate(U_i, U_j, U_ij_bar, specific_entropy_j, col_idx == 0);
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
     * Constructor taking a ProblemDescription instance as argument
     */
    Limiter(const ProblemDescription &problem_description)
        : problem_description(problem_description)
    {
    }

    /**
     * Reset temporary storage
     */
    void reset(const Number specific_entropy_i);

    /**
     * When looping over the sparsity row, add the contribution associated
     * with the neighboring state U_j.
     */
    void accumulate(const state_type &U_i,
                    const state_type &U_j,
                    const state_type &U_ij_bar,
                    const Number beta_ij,
                    const Number specific_entropy_j);

    /**
     * Apply relaxation.
     */
    void apply_relaxation(const Number hd_i);

    /**
     * Return the computed bounds.
     */
    const Bounds &bounds() const;


    //*}
    /** @name */
    //@{

    /**
     * Given a state \f$\mathbf U\f$ and an update \f$\mathbf P\f$ this
     * function computes and returns the maximal coefficient \f$t\f$,
     * obeying \f$t_{\text{min}} < t < t_{\text{max}}\f$, such that the
     * selected local minimum principles are obeyed.
     */
    static std::tuple<Number, bool>
    limit(const ProblemDescription &problem_description,
          const Bounds &bounds,
          const state_type &U,
          const state_type &P,
          const ScalarNumber newton_tolerance,
          const unsigned int newton_max_iter,
          const Number t_min = Number(0.),
          const Number t_max = Number(1.));
    //*}

  private:
    /** @name */
    //@{

    const ProblemDescription &problem_description;

    Bounds bounds_;

    Number rho_relaxation_numerator;
    Number rho_relaxation_denominator;

    Number s_interp_max;

    //@}
  };


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void
  Limiter<dim, Number>::reset(const Number specific_entropy_i)
  {
    auto &[rho_min, rho_max, s_min] = bounds_;

    rho_min = Number(std::numeric_limits<ScalarNumber>::max());
    rho_max = Number(0.);

    rho_relaxation_numerator = Number(0.);
    rho_relaxation_denominator = Number(0.);

    s_min = specific_entropy_i;
    s_interp_max = Number(0.);
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void
  Limiter<dim, Number>::accumulate(const state_type &U_i,
                                   const state_type &U_j,
                                   const state_type &U_ij_bar,
                                   const Number beta_ij,
                                   const Number specific_entropy_j)
  {
    /* Bounds: */

    auto &[rho_min, rho_max, s_min] = bounds_;

    const auto rho_ij = problem_description.density(U_ij_bar);
    rho_min = std::min(rho_min, rho_ij);
    rho_max = std::max(rho_max, rho_ij);

    s_min = std::min(s_min, specific_entropy_j);

    /* Relaxation: */

    const auto rho_i = problem_description.density(U_i);
    const auto rho_j = problem_description.density(U_j);
    rho_relaxation_numerator += beta_ij * (rho_i + rho_j);
    rho_relaxation_denominator += beta_ij;

    const Number s_interp =
        problem_description.specific_entropy((U_i + U_j) * ScalarNumber(.5));
    s_interp_max = std::max(s_interp_max, s_interp);
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void
  Limiter<dim, Number>::apply_relaxation(Number hd_i)
  {
    auto &[rho_min, rho_max, s_min] = bounds_;

    constexpr unsigned int relaxation_order_ = 3;
    const Number r_i =
        Number(2.) * dealii::Utilities::fixed_power<relaxation_order_>(
                         std::sqrt(std::sqrt(hd_i)));

    constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
    const Number rho_relaxation =
        std::abs(rho_relaxation_numerator) /
        (std::abs(rho_relaxation_denominator) + Number(eps));

    rho_min = std::max((Number(1.) - r_i) * rho_min, rho_min - rho_relaxation);
    rho_max = std::min((Number(1.) + r_i) * rho_max, rho_max + rho_relaxation);

    s_min =
        std::max((Number(1.) - r_i) * s_min, Number(2.) * s_min - s_interp_max);
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline const typename Limiter<dim, Number>::Bounds &
  Limiter<dim, Number>::bounds() const
  {
    return bounds_;
  }

} /* namespace ryujin */
