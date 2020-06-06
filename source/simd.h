//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef SIMD_H
#define SIMD_H

#include <compile_time_options.h>

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

namespace ryujin
{
  template<int dim , typename Number> class ProblemDescription;

  /**
   * @name Type traits and packed index handling
   */
  //@{

  /**
   * Small helper class to extract the underlying scalar type of a
   * VectorizedArray, or return T directly.
   *
   * @ingroup SIMD
   */
  //@{
  template <typename T>
  struct get_value_type {
    using type = T;
  };


  template <typename T, std::size_t width>
  struct get_value_type<dealii::VectorizedArray<T, width>> {
    using type = T;
  };
  //@}

#ifndef DOXYGEN
  namespace
  {
    template <typename Functor, size_t... Is>
    auto generate_iterators_impl(Functor f, std::index_sequence<Is...>)
        -> std::array<decltype(f(0)), sizeof...(Is)>
    {
      return {f(Is)...};
    }
  } /* namespace */
#endif

  /**
   * Given a callable object f(k), this function creates a std::array with
   * elements initialized as follows:
   *
   *   { f(0) , f(1) , ... , f(length - 1) }
   *
   * We use this function to create an array of sparsity iterators that
   * cannot be default initialized.
   *
   * @ingroup SIMD
   */
  template <unsigned int length, typename Functor>
  DEAL_II_ALWAYS_INLINE inline auto generate_iterators(Functor f)
      -> std::array<decltype(f(0)), length>
  {
    return generate_iterators_impl<>(f, std::make_index_sequence<length>());
  }


  /**
   * Increment all iterators in an std::array simultaneously.
   *
   * @ingroup SIMD
   */
  template <typename T>
  DEAL_II_ALWAYS_INLINE inline void increment_iterators(T &iterators)
  {
    for (auto &it : iterators)
      it++;
  }

  //@}
  /**
   * @name Transcendental and other mathematical operations
   */
  //@{

  /**
   * Return the positive part of a number.
   *
   * @ingroup SIMD
   */
  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE Number positive_part(const Number number)
  {
    return Number(0.5) * (std::abs(number) + number);
  }


  /**
   * Return the negative part of a number.
   *
   * @ingroup SIMD
   */
  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE Number negative_part(const Number number)
  {
    return Number(0.5) * (std::abs(number) - number);
  }


  /**
   * A wrapper around dealii::Utilities::fixed_power. We use a wrapper
   * instead of calling the function directly so that we can easily change
   * the implementation at one central place.
   *
   * @ingroup SIMD
   */
  template <int N, typename T>
  inline T fixed_power(const T x)
  {
    return dealii::Utilities::fixed_power<N, T>(x);
  }


  /**
   * Custom implementation of a vectorized pow function.
   *
   * @ingroup SIMD
   */
  template <typename T>
  T pow(const T x, const typename get_value_type<T>::type b);

  //@}
  /**
   * @name SIMD based access to vectors and arrays of vectors
   */
  //@{

#ifndef DOXYGEN
  namespace
  {
    // Some deal.II classes make the underlying data available via data(),
    // others via get_values(). Let's provide a uniform access helper.

    template <typename T>
    DEAL_II_ALWAYS_INLINE inline auto access_data(T &U)
        -> decltype(U.get_values())
    {
      return U.get_values();
    }

    template <typename T>
    DEAL_II_ALWAYS_INLINE inline auto access_data(T &U) -> decltype(U.data())
    {
      return U.data();
    }
  } /* namespace */
#endif

  /**
   * Return a VectorizedArray with
   *   { U[i] , U[i + 1] , ... , U[i + VectorizedArray::size() - 1] }
   *
   * @ingroup SIMD
   */
  template <typename T1>
  DEAL_II_ALWAYS_INLINE inline dealii::VectorizedArray<typename T1::value_type>
  simd_load(const T1 &vector, unsigned int i)
  {
    dealii::VectorizedArray<typename T1::value_type> result;
    result.load(access_data(vector) + i);
    return result;
  }


  /**
   * Return a VectorizedArray with
   *   { U[js[0] , U[js[1]] , ... , U[js[VectorizedArray::size() - 1]] }
   *
   * @ingroup SIMD
   */
  template <typename T1>
  DEAL_II_ALWAYS_INLINE inline dealii::VectorizedArray<typename T1::value_type>
  simd_load(const T1 &vector, const unsigned int *js)
  {
    dealii::VectorizedArray<typename T1::value_type> result;
    result.gather(access_data(vector), js);
    return result;
  }


  /**
   * Write out the given VectorizedArray to the vector
   *
   * @ingroup SIMD
   */
  template <typename T1>
  DEAL_II_ALWAYS_INLINE inline void
  simd_store(T1 &vector,
             const dealii::VectorizedArray<typename T1::value_type> &values,
             unsigned int i)
  {
    values.store(access_data(vector) + i);
  }

  //@}

} // namespace ryujin

#endif /* SIMD_H */
