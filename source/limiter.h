//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef LIMITER_H
#define LIMITER_H

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
   * @cite GuermondEtAl2018 and @cite KronbichlerMaier2020. Given a
   * computed set of bounds and an update direction \f$\mathbf P_{ij}\f$
   * one can now determine a candidate \f$\tilde l_{ij}\f$ by computing
   *
   * \f{align}
   *   \tilde l_{ij} = \max_{l\,\in\,[0,1]}
   *   \,\Big\{\rho_{\text{min}}\,\le\,\rho\,(\mathbf U_i +\tilde l_{ij}\mathbf P_{ij})
   *   \,\le\,\rho_{\text{max}},\quad
   *   \phi_{\text{min}}\,\le\,\phi\,(\mathbf U_{i}+\tilde l_{ij}\mathbf P_{ij})\Big\},
   * \f}
   *
   * where \f$\psi\f$ denots the specific entropy @cite KronbichlerMaier2020.
   *
   * Algorithmically this is accomplished as follows: Given an initial
   * interval \f$[t_L,t_R]\f$, where \f$t_L\f$ is a good state, we first
   * make the interval smaller ensuring the bounds on the density are
   * fulfilled. If limiting on the specific entropy is selected we then
   * then perform a quadratic Newton iteration (updating \f$[t_L,t_R]\f$
   * solving for the root of a 3-convex function
   * \f{align}
   *     \Psi(\mathbf U)\;=\;\rho^{\gamma+1}(\mathbf U)\,\big(\phi(\mathbf U)-\phi_{\text{min}}\big).
   * \f}
   *
   * @todo document local entropy inequality condition.
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
    static constexpr unsigned int problem_dimension = ProblemDescription<dim>::problem_dimension;
    // clang-format on

    /**
     * @copydoc ProblemDescription::rank1_type
     */
    using rank1_type = typename ProblemDescription<dim>::template rank1_type<Number>;

    /**
     * @copydoc ProblemDescription::ScalarNumber
     */
    using ScalarNumber = typename get_value_type<Number>::type;

    /**
     * An enum describing the thermodynamical quantities for which the
     * invariant domain property is enforced by the limiter.
     */
    enum class Limiters {
      /** Do not limit and accept full high-order update. */
      none,
      /** Enforce local bounds on density. */
      rho,
      /** Enforce local bounds on density and specific entropy. */
      specific_entropy,
      /**
       * Enforce local bounds on density, specific entropy and enforce an
       * entropy inequality using the Harten-type inequality
       * ProblemDescription::harten_entropy().
       */
      entropy_inequality
    };

    /**
     * Constructor.
     */
    Limiter();

    /**
     * @name Limiter compile time options
     */
    //@{

    // clang-format off
    /**
     * Selected final limiting stage.
     * @ingroup CompileTimeOptions
     */
    static constexpr Limiters limiter_ = LIMITER;

    /**
     * Relax accumulated limiter bounds.
     * @ingroup CompileTimeOptions
     */
    static constexpr bool relax_bounds_ = LIMITER_RELAX_BOUNDS;

    /**
     * Order of mesh-size dependent coefficient in relaxation window.
     * @ingroup CompileTimeOptions
     */
    static constexpr unsigned int relaxation_order_ = LIMITER_RELAXATION_ORDER;

    // clang-format on

    //@}
    /**
     * @name Stencil-based accumulations of bounds
     *
     * Intended usage:
     * ```
     * Limiter<dim, Number> limiter;
     * for (unsigned int i = n_internal; i < n_owned; ++i) {
     *   // ...
     *   limiter.reset(variations_i);
     *   for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
     *     // ...
     *     limiter.accumulate(U_i, U_j, U_ij_bar, entropy_j, col_idx == 0);
     *   }
     *   limiter.apply_relaxation(hd_i);
     *   limiter_serial.bounds();
     * }
     * ```
     */
    //@{

    /**
     * The number of stored entries in the bounds array.
     *
     * @todo determine number of bounds based on chosen limiter.
     */
    // clang-format off
    static constexpr unsigned int n_bounds =
          (limiter_ == Limiters::rho) ? 2
        : (limiter_ == Limiters::specific_entropy) ? 3
        : (limiter_ == Limiters::entropy_inequality) ? 5 : 0;
    // clang-format on

    /**
     * Array type used to store accumulated bounds.
     */
    using Bounds = std::array<Number, n_bounds>;

    /**
     * Constructor taking a ProblemDescription instance as argument
     */
    Limiter(const ProblemDescription<dim, ScalarNumber> &problem_description)
        : problem_description(problem_description)
    {
    }

    /**
     * Reset temporary storage and reinitialize variations for new index i.
     */
    void reset(const Number variations_i);

    /**
     * When looping over the sparsity row, add the contribution associated
     * with the neighboring state U_j.
     */
    void accumulate(const rank1_type &U_i,
                    const rank1_type &U_j,
                    const rank1_type &U_ij_bar,
                    const Number beta_ij,
                    const Number entropy_j,
                    const Number variations_j,
                    const bool is_diagonal_entry);

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
    template <Limiters limiter = limiter_, typename BOUNDS>
    static Number
    limit(const ProblemDescription<dim, ScalarNumber> &problem_description,
          const BOUNDS &bounds,
          const rank1_type &U,
          const rank1_type &P,
          const Number t_min = Number(0.),
          const Number t_max = Number(1.));
    //*}

  private:
    /** @name */
    //@{

    const ProblemDescription<dim, ScalarNumber> &problem_description;

    Bounds bounds_;

    Number variations_i;
    Number rho_relaxation_numerator;
    Number rho_relaxation_denominator;

    Number s_interp_max;

    //@}
  };


  template <int dim, typename Number>
  Limiter<dim, Number>::Limiter()
  {
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void
  Limiter<dim, Number>::reset(const Number new_variations_i)
  {
    if constexpr (relax_bounds_) {
      variations_i = new_variations_i;
    }

    auto &[rho_min, rho_max, s_min] = bounds_;

    if constexpr (limiter_ == Limiters::none)
      return;

    rho_min = Number(std::numeric_limits<ScalarNumber>::max());
    rho_max = Number(0.);

    rho_relaxation_numerator = Number(0.);
    rho_relaxation_denominator = Number(0.);

    if constexpr (limiter_ == Limiters::specific_entropy) {
      s_min = Number(std::numeric_limits<ScalarNumber>::max());
      s_interp_max = Number(0.);
    }
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void
  Limiter<dim, Number>::accumulate(const rank1_type &U_i,
                                   const rank1_type &U_j,
                                   const rank1_type &U_ij_bar,
                                   const Number beta_ij,
                                   const Number entropy_j,
                                   const Number variations_j,
                                   const bool is_diagonal_entry)
  {
    /* Relaxation (the numerical constant 8 is up to debate): */
    if constexpr (relax_bounds_) {
      rho_relaxation_numerator +=
          Number(8.0 * 0.5) * beta_ij * (variations_i + variations_j);

      rho_relaxation_denominator += beta_ij;
    }

    /* Bounds: */

    auto &[rho_min, rho_max, s_min] = bounds_;

    if constexpr (limiter_ == Limiters::none)
      return;

    const auto rho_ij = U_ij_bar[0];
    rho_min = std::min(rho_min, rho_ij);
    rho_max = std::max(rho_max, rho_ij);

    if constexpr (limiter_ == Limiters::specific_entropy) {
      s_min = std::min(s_min, entropy_j);

      if (!is_diagonal_entry) {
        const Number s_interp = problem_description.specific_entropy(
            (U_i + U_j) * ScalarNumber(.5));
        s_interp_max = std::max(s_interp_max, s_interp);
      }
    }
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void
  Limiter<dim, Number>::apply_relaxation(Number hd_i)
  {
    if constexpr (!relax_bounds_)
      return;

    auto &[rho_min, rho_max, s_min] = bounds_;

    if constexpr (limiter_ == Limiters::none)
      return;

    const Number r_i =
        Number(2.) * dealii::Utilities::fixed_power<relaxation_order_>(
                         std::sqrt(std::sqrt(hd_i)));

    constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
    const Number rho_relaxation =
        std::abs(rho_relaxation_numerator) /
        (std::abs(rho_relaxation_denominator) + Number(eps));

    rho_min = std::max((Number(1.) - r_i) * rho_min, rho_min - rho_relaxation);
    rho_max = std::min((Number(1.) + r_i) * rho_max, rho_max + rho_relaxation);

    if constexpr (limiter_ == Limiters::specific_entropy) {
      s_min = std::max((Number(1.) - r_i) * s_min,
                       Number(2.) * s_min - s_interp_max);
    }
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline const typename Limiter<dim, Number>::Bounds &
  Limiter<dim, Number>::bounds() const
  {
    return bounds_;
  }

} /* namespace ryujin */

#endif /* LIMITER_H */

#ifdef OBSESSIVE_INLINING
#include "limiter.template.h"
#endif
