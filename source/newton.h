//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "simd.h"

#include <array>

namespace ryujin
{
  /**
   * @name Functions
   */
  //@{

  /**
   * Perform one step of a quadratic Newton iteration, see
   * @cite ryujin-2021-1, Algorithm 3.
   *
   * For this, it has to hold true that \f$p_1\le p^\ast\le p_2\f$, and
   * \f$\phi(p_1)\le 0\le \phi(p_2)\f$, or \f$\phi(p_1)\ge 0\ge
   * \phi(p_2)\f$ and that \f$\phi\f$ itself is a 3-convex/concave
   * function, i.e., \f$\phi'''<0\f$, or \f$\phi'''>0\f$.
   *
   * Modifies p_1 and P_2 ensures that p_1 <= p_2, and that p_1 (p_2) is
   * monotonically increasing (decreasing).
   *
   * @todo Write out the quadratic Newton step in more detail.
   *
   * @ingroup EulerModule
   */
  template <typename Number>
  DEAL_II_ALWAYS_INLINE inline void
  quadratic_newton_step(Number &p_1,
                        Number &p_2,
                        const Number phi_p_1,
                        const Number phi_p_2,
                        const Number dphi_p_1,
                        const Number dphi_p_2,
                        const Number sign = Number(1.0))
  {
    using ScalarNumber = typename get_value_type<Number>::type;
    constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();

    /* Compute divided differences: */

    const auto scaling = ScalarNumber(1.) / (p_2 - p_1 + Number(eps));

    const Number dd_11 = dphi_p_1;
    const Number dd_12 = (phi_p_2 - phi_p_1) * scaling;
    const Number dd_22 = dphi_p_2;

    const Number dd_112 = (dd_12 - dd_11) * scaling;
    const Number dd_122 = (dd_22 - dd_12) * scaling;

    /* Update left and right point: */

    const auto discriminant_1 =
        std::abs(dphi_p_1 * dphi_p_1 - ScalarNumber(4.) * phi_p_1 * dd_112);
    const auto discriminant_2 =
        std::abs(dphi_p_2 * dphi_p_2 - ScalarNumber(4.) * phi_p_2 * dd_122);

    const auto denominator_1 = dphi_p_1 + sign * std::sqrt(discriminant_1);
    const auto denominator_2 = dphi_p_2 + sign * std::sqrt(discriminant_2);

    /* Make sure we do not produce NaNs: */

    auto t_1 =
        p_1 - dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                  std::abs(denominator_1),
                  Number(eps),
                  Number(0.),
                  ScalarNumber(2.) * phi_p_1 / denominator_1);

    auto t_2 =
        p_2 - dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                  std::abs(denominator_2),
                  Number(eps),
                  Number(0.),
                  ScalarNumber(2.) * phi_p_2 / denominator_2);

    /* Enforce bounds: */

    t_1 = std::max(p_1, t_1);
    t_2 = std::max(p_1, t_2);
    t_1 = std::min(p_2, t_1);
    t_2 = std::min(p_2, t_2);

    /* Ensure that always p_1 <= p_2: */

    p_1 = std::min(t_1, t_2);
    p_2 = std::max(t_1, t_2);

    return;
  }

  //@}

} /* namespace ryujin */
