//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#ifdef DEAL_II_WITH_GSL
#include <gsl/gsl_spline.h>

namespace ryujin
{
  /**
   * A cubic spline class implemented as a thin wrapper around the GSL
   * spline library functions.
   *
   * Usage:
   * @code
   * std::vector<double> xs{0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
   * std::vector<double> ys{1.0, 0.2, 5.0, 2.0, 1.0, 10.0};
   * CubicSpline spline(xs, ys);
   *
   * spline.eval(0.5);
   * @endcode
   *
   * @ingroup Mesh
   */
  class CubicSpline
  {
  public:
    /**
     * Constructor.
     *
     * @pre The supplied vectors @p x and @p y must have the same size and
     * must contain at least two elements. The vector @p x must be sorted
     */
    CubicSpline(const std::vector<double> &x,
                const std::vector<double> &y) noexcept
        : x_(x)
        , y_(y)
    {
      AssertNothrow(x_.size() == y_.size(), dealii::ExcInternalError());
      AssertNothrow(x_.size() >= 2, dealii::ExcInternalError());
      AssertNothrow(std::is_sorted(x_.begin(), x_.end()),
                    dealii::ExcInternalError());

      spline = gsl_spline_alloc(gsl_interp_cspline, x_.size());
      gsl_spline_init(spline, x_.data(), y_.data(), x_.size());

      accel = gsl_interp_accel_alloc();
    }

    /**
     * Copy constructor.
     */
    CubicSpline(const CubicSpline &copy)
        : CubicSpline(copy.x_, copy.y_)
    {
    }

    /**
     * The copy assignment operator is deleted.
     */
    CubicSpline &operator=(const CubicSpline &) = delete;

    /**
     * Destructor.
     */
    ~CubicSpline()
    {
      gsl_interp_accel_free(accel);
      gsl_spline_free(spline);
    }

    /**
     * Evaluate the cubic spline at a given point @p x.
     *
     * @pre The point @p x must lie within the interval described by the
     * largest and smallest support point supplied to the constructor.
     */
    inline double eval(double x) const
    {
      return gsl_spline_eval(spline, x, accel);
    }

  private:
    const std::vector<double> x_;
    const std::vector<double> y_;
    gsl_spline *spline;
    mutable gsl_interp_accel *accel;
  };
} // namespace ryujin

#endif
