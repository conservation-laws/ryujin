//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "simd.h"
#include "simd_fast_pow.template.h"

#include <cmath>

#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)
// Import the vectorlib library prefixed in the vcl namespace
#define VCL_NAMESPACE vcl
DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#include "../simd-math/vectorclass.h"
#include "../simd-math/vectormath_exp.h"
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS
#undef VCL_NAMESPACE
#else
// Make ryujin::pow known as vcl::ryujin
namespace vcl = ryujin;
#endif

namespace ryujin
{
  /*****************************************************************************
   *                Helper typetraits for dealing with vcl:                    *
   ****************************************************************************/

  namespace
  {
    /*
     * A type trait to select the correct VCL type:
     */
    template <typename T, std::size_t width>
    struct VectorClassType {
    };

#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 3 && defined(__AVX512F__)
    template <>
    struct VectorClassType<float, 16> {
      using value_type = vcl::Vec16f;
    };

    template <>
    struct VectorClassType<double, 8> {
      using value_type = vcl::Vec8d;
    };
#endif

#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 2 && defined(__AVX__)
    template <>
    struct VectorClassType<float, 8> {
      using value_type = vcl::Vec8f;
    };

    template <>
    struct VectorClassType<double, 4> {
      using value_type = vcl::Vec4d;
    };
#endif

#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)
    template <>
    struct VectorClassType<float, 4> {
      using value_type = vcl::Vec4f;
    };

    template <>
    struct VectorClassType<double, 2> {
      using value_type = vcl::Vec2d;
    };

    template <>
    struct VectorClassType<float, 1> {
      using value_type = vcl::Vec4f;
    };

    template <>
    struct VectorClassType<double, 1> {
      using value_type = vcl::Vec2d;
    };

#else
    template <>
    struct VectorClassType<float, 1> {
      using value_type = float;
    };

    template <>
    struct VectorClassType<double, 1> {
      using value_type = double;
    };
#endif

    /*
     * Convert a dealii::VectorizedArray to a VCL container type:
     */
    template <typename T, std::size_t width>
    DEAL_II_ALWAYS_INLINE inline typename VectorClassType<T, width>::value_type
    to_vcl(const dealii::VectorizedArray<T, width> x)
    {
      return typename VectorClassType<T, width>::value_type(x.data);
    }


    /*
     * Convert a VCL container type to a dealii::VectorizedArray:
     */
    template <typename T, std::size_t width>
    DEAL_II_ALWAYS_INLINE inline dealii::VectorizedArray<T, width>
    from_vcl(typename VectorClassType<T, width>::value_type x)
    {
      dealii::VectorizedArray<T, width> result;
#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)
      if constexpr (width == 1)
        result.data = x.extract(0);
      else
#endif
        result.data = x;
      return result;
    }


    /*
     * Helper functions to convert to float arrays and back.
     */
    template <typename T, std::size_t width>
    struct FC {
    };

    template <std::size_t width>
    struct FC<double, width> {
      // There is no Vec2f, so make sure to use Vec4f instead
#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)
      static constexpr std::size_t float_width = (width <= 2 ? 4 : width);
#else
      static_assert(width == 1, "internal error");
      static constexpr std::size_t float_width = width;
#endif

      static DEAL_II_ALWAYS_INLINE inline
          typename VectorClassType<float, float_width>::value_type
          to_float(typename VectorClassType<double, width>::value_type x)
      {
#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)
        return vcl::to_float(x);
#else
        return x;
#endif
      }

      static DEAL_II_ALWAYS_INLINE inline
          typename VectorClassType<double, width>::value_type
          to_double(typename VectorClassType<float, float_width>::value_type x)
      {
#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)
        if constexpr (width == 1) {
          return static_cast<double>(x.extract(0));
        } else if constexpr (width == 2) {
          const vcl::Vec4d temp = vcl::to_double(x);
          return vcl::Vec2d(temp.extract(0), temp.extract(1));
        } else {
          return vcl::to_double(x);
        }
#else
        return x;
#endif
      }
    };

    template <std::size_t width>
    struct FC<float, width> {
      static DEAL_II_ALWAYS_INLINE inline
          typename VectorClassType<float, width>::value_type
          to_float(typename VectorClassType<float, width>::value_type x)
      {
        return x;
      }
      static DEAL_II_ALWAYS_INLINE inline
          typename VectorClassType<float, width>::value_type
          to_double(typename VectorClassType<float, width>::value_type x)
      {
        return x;
      }
    };
  } // namespace


  /*****************************************************************************
   *                           pow() implementation:                           *
   ****************************************************************************/

#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)
  template <>
  // DEAL_II_ALWAYS_INLINE inline
  float pow(const float x, const float b)
  {
    /* Use a custom pow implementation instead of std::pow(): */
    return vcl::pow(vcl::Vec4f(x), b).extract(0);
  }


  template <>
  // DEAL_II_ALWAYS_INLINE inline
  double pow(const double x, const double b)
  {
    /* Use a custom pow implementation instead of std::pow(): */
    return vcl::pow(vcl::Vec2d(x), b).extract(0);
  }


#else
  template <>
  // DEAL_II_ALWAYS_INLINE inline
  float pow(const float x, const float b)
  {
    // Call generic std::pow() implementation
    return std::pow(x, b);
  }


  template <>
  // DEAL_II_ALWAYS_INLINE inline
  double pow(const double x, const double b)
  {
    // Call generic std::pow() implementation
    return std::pow(x, b);
  }
#endif


  template <typename T, std::size_t width>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<T, width>
  pow(const dealii::VectorizedArray<T, width> x, const T b)
  {
    return from_vcl<T, width>(vcl::pow(to_vcl(x), b));
  }


  template <typename T, std::size_t width>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<T, width>
  pow(const dealii::VectorizedArray<T, width> x,
      const dealii::VectorizedArray<T, width> b)
  {
    return from_vcl<T, width>(vcl::pow(to_vcl(x), to_vcl(b)));
  }


  /*****************************************************************************
   *                         Fast pow() implementation:                        *
   ****************************************************************************/

#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)
  template <>
  // DEAL_II_ALWAYS_INLINE inline
  float fast_pow(const float x, const float b, const Bias bias)
  {
    /* Use a custom pow implementation instead of std::pow(): */
    return fast_pow_impl(vcl::Vec4f(x), vcl::Vec4f(b), bias).extract(0);
  }


  template <>
  // DEAL_II_ALWAYS_INLINE inline
  double fast_pow(const double x, const double b, const Bias bias)
  {
    /* Use a custom pow implementation instead of std::pow(): */
    return fast_pow_impl(vcl::Vec4f(x), vcl::Vec4f(b), bias).extract(0);
  }


#else
  template <>
  // DEAL_II_ALWAYS_INLINE inline
  float fast_pow(const float x, const float b, const Bias)
  {
    // Call generic std::pow() implementation
    return std::pow(x, b);
  }


  template <>
  // DEAL_II_ALWAYS_INLINE inline
  double fast_pow(const double x, const double b, const Bias)
  {
    // Call generic std::pow() implementation
    return std::pow(static_cast<float>(x), static_cast<float>(b));
  }
#endif


  template <typename T, std::size_t width>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<T, width> fast_pow(
      const dealii::VectorizedArray<T, width> x, const T b, const Bias bias)
  {
    using vcl_type = decltype(FC<T, width>::to_float(to_vcl(x)));
    return from_vcl<T, width>(FC<T, width>::to_double(
        fast_pow_impl(FC<T, width>::to_float(to_vcl(x)), vcl_type(b), bias)));
  }


  template <typename T, std::size_t width>
  // DEAL_II_ALWAYS_INLINE inline
  dealii::VectorizedArray<T, width>
  fast_pow(const dealii::VectorizedArray<T, width> x,
           const dealii::VectorizedArray<T, width> b,
           const Bias bias)
  {
    return from_vcl<T, width>(
        FC<T, width>::to_double(fast_pow_impl(FC<T, width>::to_float(to_vcl(x)),
                                              FC<T, width>::to_float(to_vcl(b)),
                                              bias)));
  }

} // namespace ryujin
