//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef CUBIC_SPLINE_H
#define CUBIC_SPLINE_H

#ifdef DEAL_II_WITH_GSL
#include <gsl/gsl_spline.h>

namespace ryujin
{
  /**
   * @todo Documentation
   */
  class CubicSpline
  {
  public:
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

    CubicSpline(const CubicSpline &copy)
        : CubicSpline(copy.x_, copy.y_)
    {
    }

    CubicSpline& operator=(const CubicSpline& copy) = delete;

    ~CubicSpline()
    {
      gsl_interp_accel_free(accel);
      gsl_spline_free(spline);
    }

    DEAL_II_ALWAYS_INLINE inline double eval(double x) const
    {
      return gsl_spline_eval(spline, x, accel);
    }

  private:
    const std::vector<double> x_;
    const std::vector<double> y_;
    gsl_spline *spline;
    mutable gsl_interp_accel *accel;
  };
}

#endif

#endif /* CUBIC_SPLINE_H */
