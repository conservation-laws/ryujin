#ifndef SIMD_H
#define SIMD_H

#include <deal.II/base/vectorization.h>
#include <deal.II/base/utilities.h>

DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#include "../simd-math/vectorclass.h"
#include "../simd-math/vectormath_exp.h"
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS

namespace grendel
{
  /*
   * A wrapper around dealii::Utilities::fixed_power. We use a wrapper
   * instead of calling the function directly so that we can easily change
   * the implementation at one central place.
   */
  template <int N, typename T>
  DEAL_II_ALWAYS_INLINE inline T fixed_power(const T x)
  {
    return dealii::Utilities::fixed_power<N, T>(x);
  }


  namespace
  {
    /*
     * Small helper class to extract the underlying scalar type of a
     * VectorizedArray, or return T directly.
     */
    template <typename T>
    struct get_value_type {
      using type = T;
    };

    template <typename T, int width>
    struct get_value_type<dealii::VectorizedArray<T, width>> {
      using type = T;
    };
  } // namespace


  template <typename T>
  inline T pow(const T x, const typename get_value_type<T>::type b) = delete;

#  if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__) && defined(USE_CUSTOM_POW)

  template<>
  DEAL_II_ALWAYS_INLINE inline float pow(const float x, const float b)
  {
    /* Use a custom pow implementation for SIMD vector units: */
    return pow(Vec4f(x), b).extract(0);
  }

  template <>
  DEAL_II_ALWAYS_INLINE inline double pow(const double x, const double b)
  {
    /* Use a custom pow implementation for SIMD vector units: */
    return pow(Vec2d(x), b).extract(0);
  }


#else

  template <>
  DEAL_II_ALWAYS_INLINE inline float pow(const float x, const float b)
  {
    // Call generic implementation
    return std::pow(x, b);
  }

  template <>
  DEAL_II_ALWAYS_INLINE inline double pow(const double x, const double b)
  {
    // Call generic implementation
    return std::pow(x, b);
  }
#endif

#  if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 3 && defined(__AVX512F__)

  template <>
  DEAL_II_ALWAYS_INLINE inline dealii::VectorizedArray<float, 16>
  pow(const dealii::VectorizedArray<float, 16> x, const float b)
  {
    dealii::VectorizedArray<float, 16> result;
    result.data = pow(Vec16f(x.data), b);
    return result;
  }

  template <>
  DEAL_II_ALWAYS_INLINE inline dealii::VectorizedArray<double, 8>
  pow(const dealii::VectorizedArray<double, 8> x, const double b)
  {
    dealii::VectorizedArray<double, 8> result;
    result.data =  pow(Vec8d(x.data), b);
    return result;
  }

#  endif

#  if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 2 && defined(__AVX__)

  template <>
  DEAL_II_ALWAYS_INLINE inline dealii::VectorizedArray<float, 8>
  pow(const dealii::VectorizedArray<float, 8> x, const float b)
  {
    dealii::VectorizedArray<float, 8> result;
    result.data =  pow(Vec8f(x.data), b);
    return result;
  }

  template <>
  DEAL_II_ALWAYS_INLINE inline dealii::VectorizedArray<double, 4>
  pow(const dealii::VectorizedArray<double, 4> x, const double b)
  {
    dealii::VectorizedArray<double, 4> result;
    result.data =  pow(Vec4d(x.data), b);
    return result;
  }

#  endif

#  if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)

  template <>
  DEAL_II_ALWAYS_INLINE inline dealii::VectorizedArray<float, 4>
  pow(const dealii::VectorizedArray<float, 4> x, const float b)
  {
    dealii::VectorizedArray<float, 4> result;
    result.data =  pow(Vec4f(x.data), b);
    return result;
  }

  template <>
  DEAL_II_ALWAYS_INLINE inline dealii::VectorizedArray<double, 2>
  pow(const dealii::VectorizedArray<double, 2> x, const double b)
  {
    dealii::VectorizedArray<double, 2> result;
    result.data =  pow(Vec2d(x.data), b);
    return result;
  }

#  endif


} // namespace grendel

#endif /* SIMD_H */
