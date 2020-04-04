#ifndef SIMD_H
#define SIMD_H

#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

namespace grendel
{
  /**
   * Return the positive part of a number.
   */
  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE Number positive_part(const Number number)
  {
    return Number(0.5) * (std::abs(number) + number);
  }


  /**
   * Return the negative part of a number.
   */
  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE Number negative_part(const Number number)
  {
    return Number(0.5) * (std::abs(number) - number);
  }


  /*
   * Small helper class to extract the underlying scalar type of a
   * VectorizedArray, or return T directly.
   */
  template <typename T>
  struct get_value_type {
    using type = T;
  };

  template <typename T, std::size_t width>
  struct get_value_type<dealii::VectorizedArray<T, width>> {
    using type = T;
  };


  /*
   * A wrapper around dealii::Utilities::fixed_power. We use a wrapper
   * instead of calling the function directly so that we can easily change
   * the implementation at one central place.
   */
  template <int N, typename T>
  inline T fixed_power(const T x)
  {
    return dealii::Utilities::fixed_power<N, T>(x);
  }


  /*
   * Custom implementation of a vectorized pow function
   */
  template <typename T>
  T pow(const T x, const typename get_value_type<T>::type b);

} // namespace grendel

#endif /* SIMD_H */
