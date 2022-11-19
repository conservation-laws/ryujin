//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "simd.h"

DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)
#define VCL_NAMESPACE vcl
#include "../simd-math/vectorclass.h"
#include "../simd-math/vectormath_exp.h"
#undef VCL_NAMESPACE
#endif
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS

namespace ryujin
{
#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__) &&          \
    defined(WITH_CUSTOM_POW)

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  float pow(const float x, const float b)
  {
    /* Use a custom pow implementation for SIMD vector units: */
    return vcl::pow(vcl::Vec4f(x), b).extract(0);
  }


  template <>
  // DEAL_II_ALWAYS_INLINE inline
  double pow(const double x, const double b)
  {
    /* Use a custom pow implementation for SIMD vector units: */
    return vcl::pow(vcl::Vec2d(x), b).extract(0);
  }


#else

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  float pow(const float x, const float b)
  {
    // Call generic implementation
    return std::pow(x, b);
  }

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  double pow(const double x, const double b)
  {
    // Call generic implementation
    return std::pow(x, b);
  }

#endif

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<float, 1>
  pow(const dealii::VectorizedArray<float, 1> x, const float b)
  {
    return ryujin::pow(x.data, b);
  }

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<double, 1>
  pow(const dealii::VectorizedArray<double, 1> x, const double b)
  {
    return ryujin::pow(x.data, b);
  }

#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 3 && defined(__AVX512F__)

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<float, 16>
  pow(const dealii::VectorizedArray<float, 16> x, const float b)
  {
    dealii::VectorizedArray<float, 16> result;
    result.data = vcl::pow(vcl::Vec16f(x.data), b);
    return result;
  }

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<float, 16>
  pow(const dealii::VectorizedArray<float, 16> x,
      const dealii::VectorizedArray<float, 16> b)
  {
    dealii::VectorizedArray<float, 16> result;
    result.data = vcl::pow(vcl::Vec16f(x.data), vcl::Vec16f(b.data));
    return result;
  }

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<double, 8>
  pow(const dealii::VectorizedArray<double, 8> x, const double b)
  {
    dealii::VectorizedArray<double, 8> result;
    result.data = vcl::pow(vcl::Vec8d(x.data), b);
    return result;
  }

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<double, 8>
  pow(const dealii::VectorizedArray<double, 8> x,
      const dealii::VectorizedArray<double, 8> b)
  {
    dealii::VectorizedArray<double, 8> result;
    result.data = vcl::pow(vcl::Vec8d(x.data), vcl::Vec8d(b.data));
    return result;
  }

#endif

#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 2 && defined(__AVX__)

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<float, 8>
  pow(const dealii::VectorizedArray<float, 8> x, const float b)
  {
    dealii::VectorizedArray<float, 8> result;
    result.data = vcl::pow(vcl::Vec8f(x.data), b);
    return result;
  }

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<float, 8>
  pow(const dealii::VectorizedArray<float, 8> x,
      const dealii::VectorizedArray<float, 8> b)
  {
    dealii::VectorizedArray<float, 8> result;
    result.data = vcl::pow(vcl::Vec8f(x.data), vcl::Vec8f(b.data));
    return result;
  }

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<double, 4>
  pow(const dealii::VectorizedArray<double, 4> x, const double b)
  {
    dealii::VectorizedArray<double, 4> result;
    result.data = vcl::pow(vcl::Vec4d(x.data), b);
    return result;
  }

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<double, 4>
  pow(const dealii::VectorizedArray<double, 4> x,
      const dealii::VectorizedArray<double, 4> b)
  {
    dealii::VectorizedArray<double, 4> result;
    result.data = vcl::pow(vcl::Vec4d(x.data), vcl::Vec4d(b.data));
    return result;
  }

#endif

#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<float, 4>
  pow(const dealii::VectorizedArray<float, 4> x, const float b)
  {
    dealii::VectorizedArray<float, 4> result;
    result.data = vcl::pow(vcl::Vec4f(x.data), b);
    return result;
  }

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<float, 4>
  pow(const dealii::VectorizedArray<float, 4> x,
      const dealii::VectorizedArray<float, 4> b)
  {
    dealii::VectorizedArray<float, 4> result;
    result.data = vcl::pow(vcl::Vec4f(x.data), vcl::Vec4f(b.data));
    return result;
  }

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<double, 2>
  pow(const dealii::VectorizedArray<double, 2> x, const double b)
  {
    dealii::VectorizedArray<double, 2> result;
    result.data = vcl::pow(vcl::Vec2d(x.data), b);
    return result;
  }

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<double, 2>
  pow(const dealii::VectorizedArray<double, 2> x,
      const dealii::VectorizedArray<double, 2> b)
  {
    dealii::VectorizedArray<double, 2> result;
    result.data = vcl::pow(vcl::Vec2d(x.data), vcl::Vec2d(b.data));
    return result;
  }

#endif

} // namespace ryujin
