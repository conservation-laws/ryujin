//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/discretization/time_integrator.h>

#include <deal.II/base/exceptions.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ryujin
{
  /**
   * The Butcher tableau of an explicit, or an implicit-explicit Runge-Kutta
   * scheme with \f$s\f$ stages.
   *
   * The tableau is stored row by row. The first row belongs to the state
   * at the beginning of the time step, the rows \f$2,\ldots,s\f$ belong
   * to the intermediate stages, and the last row \f$s+1\f$ contains the
   * weights of the final update. Trailing zeros of a row can be omitted.
   *
   * The explicit tableau @p a_explicit is strictly lower triangular. For
   * an implicit-explicit scheme the implicit tableau @p a_implicit is
   * lower triangular and shares the coefficients @p c with the explicit
   * tableau; for an explicit scheme @p a_implicit is empty.
   *
   * @ingroup MeshAndOfflineData
   */
  template <typename Number = double>
  struct ButcherTableau {
    using Matrix = std::vector<std::vector<Number>>;

    std::vector<Number> c;
    Matrix a_explicit;
    Matrix a_implicit;

    /**
     * Returns true if the tableau describes an implicit-explicit scheme.
     */
    bool is_imex() const
    {
      return !a_implicit.empty();
    }

    /**
     * Return the entry \f$a_{l,k}\f$ of the tableau @p a (with zero based
     * indices), or zero if the entry is not stored.
     */
    static Number
    entry(const Matrix &a, const std::size_t l, const std::size_t k)
    {
      return (l < a.size() && k < a[l].size()) ? a[l][k] : Number(0.);
    }

    /**
     * For every stage return the (zero based) index of the stage it is
     * computed from: every stage is an increment of the closest earlier
     * stage with a strictly smaller coefficient c, see
     * @cite ErnGuermond2022, Eq. (12). The entry of the first stage is
     * unused.
     */
    std::vector<std::size_t> start_stages() const
    {
      std::vector<std::size_t> result(c.size(), 0);
      for (std::size_t l = 1; l < c.size(); ++l)
        for (std::size_t k = 1; k < l; ++k)
          if (c[k] < c[l] && c[k] >= c[result[l]])
            result[l] = k;
      return result;
    }

    /**
     * Return the largest increment \f$c_l - c_{l'}\f$ of a stage \f$l\f$
     * with respect to its start stage \f$l'\f$. The inverse is the
     * efficiency of the scheme.
     */
    Number maximal_increment() const
    {
      const auto start = start_stages();
      Number result = 0.;
      for (std::size_t l = 1; l < c.size(); ++l)
        result = std::max(result, c[l] - c[start[l]]);
      return result;
    }
  };


  /**
   * Return the Butcher tableau of the Runge-Kutta scheme @p scheme. For a
   * Strang splitting scheme the tableau of the explicit Runge-Kutta
   * scheme used for the hyperbolic subproblem is returned.
   *
   * The strong stability preserving schemes TimeSteppingScheme::ssprk_22,
   * TimeSteppingScheme::ssprk_33, and
   * TimeSteppingScheme::strang_ssprk_33_cn are implemented as convex
   * combinations of forward Euler steps and do not have a tableau.
   *
   * @ingroup MeshAndOfflineData
   */
  template <typename Number = double>
  ButcherTableau<Number> butcher_tableau(const TimeSteppingScheme scheme)
  {
    using T = ButcherTableau<Number>;

    /*
     * The tableaux are stored row by row. The first row belongs to the old
     * state and is empty, the last row contains the weights of the final
     * update; trailing zeros are omitted.
     */

    // clang-format off
    switch (scheme) {
    case TimeSteppingScheme::erk_11:
      return T{
          /* c: */
          {0., 1.},
          /* explicit: */
          {{},
           {1.}},
          /* implicit: */
          {}};

    case TimeSteppingScheme::erk_22:
      return T{
          /* c: */
          {0., 1. / 2., 1.},
          /* explicit: */
          {{},
           {1. / 2.},
           {     0., 1.}},
          /* implicit: */
          {}};

    case TimeSteppingScheme::erk_33:
      [[fallthrough]];
    case TimeSteppingScheme::strang_erk_33_cn:
      return T{
          /* c: */
          {0., 1. / 3., 2. / 3., 1.},
          /* explicit: */
          {{},
           {1. / 3.},
           {     0., 2. / 3.},
           {1. / 4.,      0., 3. / 4.}},
          /* implicit: */
          {}};

    case TimeSteppingScheme::erk_43:
      [[fallthrough]];
    case TimeSteppingScheme::strang_erk_43_cn:
      return T{
          /* c: */
          {0., 1. / 4., 1. / 2., 3. / 4., 1.},
          /* explicit: */
          {{},
           {1. / 4.},
           {     0., 1. / 2.},
           {     0., 1. / 4.,  1. / 2.},
           {     0., 2. / 3., -1. / 3., 2. / 3.}},
          /* implicit: */
          {}};

    case TimeSteppingScheme::erk_54:
      return T{
          /* c: */
          {0., 0.2, 0.4, 0.6, 0.8, 1.},
          /* explicit: */
          {{},
           {                +0.2},
           {+0.26075582269554909, +0.13924417730445096},
           {-0.25856517872570289, +0.91136274166280729, -0.05279756293710430},
           {+0.21623276431503774, +0.51534223099602405, -0.81662794199265554, +0.88505294668159373},
           {-0.10511678454691901, +0.87880047152100838, -0.58903404061484477, +0.46213380485434047, +0.35321654878641495}},
          /* implicit: */
          {}};

    case TimeSteppingScheme::imex_11:
      return T{
          /* c: */
          {0., 1.},
          /* explicit: */
          {{},
           {1.}},
          /* implicit: */
          {{},
           {0., 1.}}};

    case TimeSteppingScheme::imex_22:
      return T{
          /* c: */
          {0., 1. / 2., 1.},
          /* explicit: */
          {{},
           {1. / 2.},
           {     0., 1.}},
          /* implicit: */
          {{},
           {0., 1. / 2.},
           {0.,      1., 0.}}};

    case TimeSteppingScheme::imex_33: {
      /* gamma: */
      const Number g = Number(0.5) + std::sqrt(Number(3.0)) / Number(6.0);

      return T{
          /* c: */
          {0., 1. / 3., 2. / 3., 1.},
          /* explicit: */
          {{},
           {1. / 3.},
           {     0., 2. / 3.},
           {1. / 4.,      0., 3. / 4.}},
          /* implicit: */
          {{},
           {Number(1. / 3.) - g,                       g},
           {                  g, Number(2. / 3.) - 2 * g,       g},
           {            1. / 4.,                      0., 3. / 4., 0.}}};
    }

    case TimeSteppingScheme::gsa_imex_22: {
      /* gamma and delta: */
      const Number g = Number(1.) - std::sqrt(Number(2.)) / Number(2.);
      const Number d = Number(1.) - Number(1.) / (Number(2.) * g);

      return T{
          /* c: */
          {0., g, 1.},
          /* explicit: */
          {{},
           {g},
           {d, 1 - d}},
          /* implicit: */
          {{},
           {0.,     g},
           {0., 1 - g, g}}};
    }

    case TimeSteppingScheme::gsa_imex_43:
      return T{
          /* c: */
          {0., 1. / 2., 2. / 3., 1. / 2., 1.},
          /* explicit: */
          {{},
           {  1. / 2.},
           {11. / 18., 1. / 18.},
           {  5. / 6., -5. / 6., 1. / 2.},
           {  1. / 4.,  7. / 4., 3. / 4., -7. / 4.}},
          /* implicit: */
          {{},
           {0.,  1. / 2.},
           {0.,  1. / 6.,  1. / 2.},
           {0., -1. / 2.,  1. / 2., 1. / 2.},
           {0.,  3. / 2., -3. / 2., 1. / 2., 1. / 2.}}};

    case TimeSteppingScheme::ssprk_22:
      [[fallthrough]];
    case TimeSteppingScheme::ssprk_33:
      [[fallthrough]];
    case TimeSteppingScheme::strang_ssprk_33_cn:
      break;
    }
    // clang-format on

    AssertThrow(false,
                dealii::ExcMessage("The selected time stepping scheme is "
                                   "not described by a Butcher tableau."));
    __builtin_trap();
  }
} // namespace ryujin
