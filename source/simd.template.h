//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "simd.h"

DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)
#include "../simd-math/vectorclass.h"
#include "../simd-math/vectormath_exp.h"
#endif
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS

namespace ryujin
{
  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<float, 1>
  pow(const dealii::VectorizedArray<float, 1> x, const float b)
  {
    return std::pow(x.data, b);
  }

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<double, 1>
  pow(const dealii::VectorizedArray<double, 1> x, const double b)
  {
    return std::pow(x.data, b);
  }

#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__) &&          \
    defined(WITH_CUSTOM_POW)

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  float pow(const float x, const float b)
  {
    /* Use a custom pow implementation for SIMD vector units: */
    return pow(Vec4f(x), b).extract(0);
  }


  template <>
  // DEAL_II_ALWAYS_INLINE inline
  double pow(const double x, const double b)
  {
    /* Use a custom pow implementation for SIMD vector units: */
    return pow(Vec2d(x), b).extract(0);
  }

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  double vec_pow(const double x, const double b)
  {
    /* Use a custom pow implementation for SIMD vector units: */
    return pow(Vec2d(x), b).extract(0);
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

#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 3 && defined(__AVX512F__)

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<float, 16>
  pow(const dealii::VectorizedArray<float, 16> x, const float b)
  {
    dealii::VectorizedArray<float, 16> result;
    result.data = pow(Vec16f(x.data), b);
    return result;
  }

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<float, 16>
  vec_pow(const dealii::VectorizedArray<float, 16> x,
          const dealii::VectorizedArray<float, 16> b)
  {
    dealii::VectorizedArray<float, 16> result;
    result.data = pow(Vec16f(x.data), Vec16f(b.data));
    return result;
  }

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<double, 8>
  pow(const dealii::VectorizedArray<double, 8> x, const double b)
  {
    dealii::VectorizedArray<double, 8> result;
    result.data = pow(Vec8d(x.data), b);
    return result;
  }

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<double, 8>
  vec_pow(const dealii::VectorizedArray<double, 8> x,
          const dealii::VectorizedArray<double, 8> b)
  {
    dealii::VectorizedArray<double, 8> result;
    result.data = pow(Vec8d(x.data), Vec8d(b.data));
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
    result.data = pow(Vec8f(x.data), b);
    return result;
  }

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<float, 8>
  vec_pow(const dealii::VectorizedArray<float, 8> x,
          const dealii::VectorizedArray<float, 8> b)
  {
    dealii::VectorizedArray<float, 8> result;
    result.data = pow(Vec8f(x.data), Vec8f(b.data));
    return result;
  }

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<double, 4>
  pow(const dealii::VectorizedArray<double, 4> x, const double b)
  {
    dealii::VectorizedArray<double, 4> result;
    result.data = pow(Vec4d(x.data), b);
    return result;
  }

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<double, 4>
  vec_pow(const dealii::VectorizedArray<double, 4> x,
          const dealii::VectorizedArray<double, 4> b)
  {
    dealii::VectorizedArray<double, 4> result;
    result.data = pow(Vec4d(x.data), Vec4d(b.data));
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
    result.data = pow(Vec4f(x.data), b);
    return result;
  }

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<float, 4>
  vec_pow(const dealii::VectorizedArray<float, 4> x,
          const dealii::VectorizedArray<float, 4> b)
  {
    dealii::VectorizedArray<float, 4> result;
    result.data = pow(Vec4f(x.data), Vec4f(b.data));
    return result;
  }

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<double, 2>
  pow(const dealii::VectorizedArray<double, 2> x, const double b)
  {
    dealii::VectorizedArray<double, 2> result;
    result.data = pow(Vec2d(x.data), b);
    return result;
  }

  template <>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<double, 2>
  vec_pow(const dealii::VectorizedArray<double, 2> x,
          const dealii::VectorizedArray<double, 2> b)
  {
    dealii::VectorizedArray<double, 2> result;
    result.data = pow(Vec2d(x.data), Vec2d(b.data));
    return result;
  }

#endif

} // namespace ryujin
