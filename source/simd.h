//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

namespace ryujin
{
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


  /**
   * Return the stride size:
   *
   * @ingroup SIMD
   */
  //@{
  template <typename T>
  unsigned int get_stride_size = 1;

  template <typename T, std::size_t width>
  unsigned int get_stride_size<dealii::VectorizedArray<T, width>> = width;
  //@}


#ifndef DOXYGEN
  namespace
  {
    template <typename Functor, size_t... Is>
    auto generate_iterators_impl(Functor f, std::index_sequence<Is...>)
        -> std::array<decltype(f(0)), sizeof...(Is)>
    {
      return {{f(Is)...}};
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
   * Custom serial pow function.
   *
   * @ingroup SIMD
   */
  template <typename T>
  T pow(const T x, const T b);


  /**
   * Custom implementation of a vectorized pow function.
   *
   * @ingroup SIMD
   */
  template <typename T, std::size_t width>
  dealii::VectorizedArray<T, width>
  pow(const dealii::VectorizedArray<T, width> x, const T b);


  /**
   * Custom implementation of a vectorized pow function with vectorized
   * exponent.
   *
   * @ingroup SIMD
   */
  template <typename T, std::size_t width>
  dealii::VectorizedArray<T, width>
  pow(const dealii::VectorizedArray<T, width> x,
      const dealii::VectorizedArray<T, width> b);


  /**
   * Controls the bias of the fast_pow() functions.
   */
  enum class Bias {
    /**
     * No specific bias.
     */
    none,

    /**
     * Guarantee an upper bound, i.e., fast_pow(x,b) >= pow(x,b) provided
     * that FIXME
     */
    max,

    /**
     * Guarantee a lower bound, i.e., fast_pow(x,b) >= pow(x,b) provided
     * that FIXME
     */
    min
  };


  /**
   * Custom serial approximate pow function.
   *
   * @ingroup SIMD
   */
  template <typename T>
  T fast_pow(const T x, const T b, const Bias bias = Bias::none);


  /**
   * Custom implementation of an approximate, vectorized pow function.
   *
   * @ingroup SIMD
   */
  template <typename T, std::size_t width>
  dealii::VectorizedArray<T, width>
  fast_pow(const dealii::VectorizedArray<T, width> x,
           const T b,
           const Bias bias = Bias::none);


  /**
   * Custom implementation of an approximate, vectorized pow function with
   * vectorized exponent.
   *
   * @ingroup SIMD
   */
  template <typename T, std::size_t width>
  dealii::VectorizedArray<T, width>
  fast_pow(const dealii::VectorizedArray<T, width> x,
           const dealii::VectorizedArray<T, width> b,
           const Bias bias = Bias::none);

  //@}
  /**
   * @name SIMD based access to vectors and arrays of vectors
   */
  //@{

  /**
   * Return a VectorizedArray with
   *   { U[i] , U[i + 1] , ... , U[i + VectorizedArray::size() - 1] }
   *
   * @ingroup SIMD
   */
  template <typename T, typename V>
  DEAL_II_ALWAYS_INLINE inline T load_value(const V &vector, unsigned int i)
  {
    static_assert(std::is_same<typename get_value_type<T>::type,
                               typename V::value_type>::value,
                  "dummy type mismatch");
    T result;

    if constexpr (std::is_same<T, typename get_value_type<T>::type>::value) {
      /* Non-vectorized sequential access. */
      result = vector.local_element(i);
    } else {
      /* Vectorized fast access. index must be divisible by simd_length */
      result.load(vector.get_values() + i);
    }

    return result;
  }


  /**
   * Return a VectorizedArray with
   *   { U[js[0] , U[js[1]] , ... , U[js[VectorizedArray::size() - 1]] }
   *
   * @ingroup SIMD
   */
  template <typename T, typename V>
  DEAL_II_ALWAYS_INLINE inline T load_value(const V &vector,
                                            const unsigned int *js)
  {
    static_assert(std::is_same<typename get_value_type<T>::type,
                               typename V::value_type>::value,
                  "dummy type mismatch");
    T result;

    if constexpr (std::is_same<T, typename get_value_type<T>::type>::value) {
      /* Non-vectorized sequential access. */
      result = vector.local_element(js[0]);
    } else {
      /* Vectorized fast access. index must be divisible by simd_length */
      result.gather(vector.get_values(), js);
    }

    return result;
  }


  /**
   * Write out the given VectorizedArray to the vector
   *
   * @ingroup SIMD
   */
  template <typename T, typename V>
  DEAL_II_ALWAYS_INLINE inline void
  store_value(V &vector, const T &values, unsigned int i)
  {
    static_assert(std::is_same<typename get_value_type<T>::type,
                               typename V::value_type>::value,
                  "dummy type mismatch");

    if constexpr (std::is_same<T, typename get_value_type<T>::type>::value) {
      /* Non-vectorized sequential access. */
      vector.local_element(i) = values;
    } else {
      /* Vectorized fast access. index must be divisible by simd_length */
      values.store(vector.get_values() + i);
    }
  }


  /**
   * Return the k-th serialized component of a Tensor of VectorizedArray
   *
   * @ingroup SIMD
   */
  template <int rank, int dim, std::size_t width, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<rank, dim, Number>
  serialize_tensor(
      const dealii::Tensor<rank, dim, dealii::VectorizedArray<Number, width>>
          &vectorized,
      const unsigned int k)
  {
    Assert(k < width, dealii::ExcMessage("Index past VectorizedArray width"));
    dealii::Tensor<rank, dim, Number> result;
    if constexpr (rank == 1) {
      for (unsigned int d = 0; d < dim; ++d)
        result[d] = vectorized[d][k];
    } else {
      for (unsigned int d = 0; d < dim; ++d)
        result[d] = serialize_tensor(vectorized[d], k);
    }
    return result;
  }


  /**
   * Variant of above function for serial Tensors that simply returns the
   * given tensor.
   *
   * @ingroup SIMD
   */
  template <int rank, int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<rank, dim, Number>
  serialize_tensor(const dealii::Tensor<rank, dim, Number> &serial,
                   const unsigned int k)
  {
    (void)k;
    Assert(k == 0,
           dealii::ExcMessage(
               "The given index k must be zero for a serial tensor"));
    return serial;
  }


  /**
   * Update the the k-th serial component of a Tensor of VectorizedArray
   *
   * @ingroup SIMD
   */
  template <int rank, int dim, std::size_t width, typename Number>
  DEAL_II_ALWAYS_INLINE inline void assign_serial_tensor(
      dealii::Tensor<rank, dim, dealii::VectorizedArray<Number, width>> &result,
      const dealii::Tensor<rank, dim, Number> &serial,
      const unsigned int k)
  {
    Assert(k < width, dealii::ExcMessage("Index past VectorizedArray width"));
    if constexpr (rank == 1) {
      for (unsigned int d = 0; d < dim; ++d)
        result[d][k] = serial[d];
    } else {
      for (unsigned int d = 0; d < dim; ++d)
        assign_serial_tensor(result[d], serial[d], k);
    }
  }


  /**
   * Variant of above function for serial Tensors that simply assigns the
   * given tensor as is.
   *
   * @ingroup SIMD
   */
  template <int rank, int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void
  assign_serial_tensor(dealii::Tensor<rank, dim, Number> &result,
                       const dealii::Tensor<rank, dim, Number> &serial,
                       const unsigned int k)
  {
    (void)k;
    Assert(k == 0,
           dealii::ExcMessage(
               "The given index k must be zero for a serial tensor"));

    result = serial;
  }

  //@}

} // namespace ryujin
