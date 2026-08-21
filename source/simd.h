//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>
#include <convenience_macros.h>

#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

#include <cmath>


/**
 * @name Exception handling in SIMD context
 */
//@{


/**
 * Mixed serial/SIMD variant of the dealii AssertThrow macro. If variable
 * is just a plain double or float, then this macro defaults to a simple
 * call to `dealii::AssertThrow(condition(variable), exception)`. Otherwise
 * (if `decltype(variable)` has a subscript operator `operator[]`, the
 * `dealii::AssertThrow` macro is expanded for all components of the
 * @p variable.
 *
 * @ingroup Miscellaneous
 */
#define AssertThrowSIMD(variable, condition, exception)                        \
  if constexpr (std::is_same<                                                  \
                    typename std::remove_const<decltype(variable)>::type,      \
                    double>::value ||                                          \
                std::is_same<                                                  \
                    typename std::remove_const<decltype(variable)>::type,      \
                    float>::value) {                                           \
    AssertThrow(condition(variable), exception);                               \
  } else {                                                                     \
    for (unsigned int k = 0; k < decltype(variable)::size(); ++k) {            \
      AssertThrow(condition((variable)[k]), exception);                        \
    }                                                                          \
  }


namespace ryujin
{
  //@}
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
  constexpr unsigned int get_stride_size = 1;

  template <typename T, std::size_t width>
  constexpr unsigned int get_stride_size<dealii::VectorizedArray<T, width>> =
      width;
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
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number positive_part(const Number number)
  {
    return std::max(Number(0.), number);
  }


  /**
   * Return the negative part of a number.
   *
   * @ingroup SIMD
   */
  template <typename Number>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number negative_part(const Number number)
  {
    return -std::min(Number(0.), number);
  }


  /**
   * A wrapper around dealii::compare_and_apply_mask() for scalar number
   * types that is annotated with DEAL_II_HOST_DEVICE so that it can be
   * used in device code.
   *
   * @ingroup SIMD
   */
  template <dealii::SIMDComparison predicate, typename Number>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
  compare_and_apply_mask(const Number &left,
                         const Number &right,
                         const Number &true_value,
                         const Number &false_value)
  {
    static_assert(std::is_floating_point_v<Number>,
                  "Only scalar number types are allowed");

    bool mask = false;
    if constexpr (predicate == dealii::SIMDComparison::equal)
      mask = (left == right);
    else if constexpr (predicate == dealii::SIMDComparison::not_equal)
      mask = (left != right);
    else if constexpr (predicate == dealii::SIMDComparison::less_than)
      mask = (left < right);
    else if constexpr (predicate == dealii::SIMDComparison::less_than_or_equal)
      mask = (left <= right);
    else if constexpr (predicate == dealii::SIMDComparison::greater_than)
      mask = (left > right);
    else
      mask = (left >= right);

    return mask ? true_value : false_value;
  }


  /**
   * Variant of above function for VectorizedArray that simply forwards to
   * the dealii::compare_and_apply_mask() implementation. (Host only.)
   *
   * @ingroup SIMD
   */
  template <dealii::SIMDComparison predicate, typename T, std::size_t width>
  DEAL_II_ALWAYS_INLINE inline dealii::VectorizedArray<T, width>
  compare_and_apply_mask(const dealii::VectorizedArray<T, width> &left,
                         const dealii::VectorizedArray<T, width> &right,
                         const dealii::VectorizedArray<T, width> &true_value,
                         const dealii::VectorizedArray<T, width> &false_value)
  {
    return dealii::compare_and_apply_mask<predicate>(
        left, right, true_value, false_value);
  }


  /**
   * A wrapper around dealii::Utilities::fixed_power. We use a wrapper
   * instead of calling the function directly so that we can easily change
   * the implementation at one central place.
   *
   * @ingroup SIMD
   */
  template <int N, typename T>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE T fixed_power(const T x)
  {
    return dealii::Utilities::fixed_power<N, T>(x);
  }


  /**
   * Custom serial pow function.
   *
   * @ingroup SIMD
   */
  template <typename T>
  DEAL_II_HOST_DEVICE T pow(const T x, const T b);


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


  template <>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE float pow(const float x, const float b)
  {
#ifdef RYUJIN_DEVICE_COMPILATION_PASS
    /* Call generic std::pow() implementation: */
    return std::pow(x, b);
#elif DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)
    /* Use a custom pow implementation instead of std::pow(): */
    return pow(dealii::VectorizedArray<float, 4>(x), b)[0];
#else
    /* Call generic std::pow() implementation: */
    return std::pow(x, b);
#endif
  }


  template <>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE double pow(const double x, const double b)
  {
#ifdef RYUJIN_DEVICE_COMPILATION_PASS
    /* Call generic std::pow() implementation */
    return std::pow(x, b);
#elif DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)
    /* Use a custom pow implementation instead of std::pow(): */
    return pow(dealii::VectorizedArray<double, 2>(x), b)[0];
#else
    /* Call generic std::pow() implementation */
    return std::pow(x, b);
#endif
  }


  /**
   * Controls the bias of the fast_pow() functions.
   */
  enum class Bias {
    /**
     * No specific bias.
     */
    none,

    /**
     * Guarantee an upper bound, i.e., fast_pow(x,b) >= pow(x,b).
     */
    max,

    /**
     * Guarantee a lower bound, i.e., fast_pow(x,b) >= pow(x,b).
     */
    min
  };


  /**
   * Custom serial approximate pow function.
   *
   * @ingroup SIMD
   */
  template <typename T>
  DEAL_II_HOST_DEVICE T fast_pow(const T x,
                                 const T b,
                                 const Bias bias = Bias::none);


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


  template <>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE float
  fast_pow(const float x, const float b, [[maybe_unused]] const Bias bias)
  {
#ifdef RYUJIN_DEVICE_COMPILATION_PASS
    /* Call generic std::pow() implementation */
    return std::pow(x, b);
#elif DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)
    /* Use a custom fast_pow implementation instead of std::pow(): */
    return fast_pow(dealii::VectorizedArray<float, 4>(x), b, bias)[0];
#else
    /* Call generic std::pow() implementation */
    return std::pow(x, b);
#endif
  }


  template <>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE double
  fast_pow(const double x, const double b, [[maybe_unused]] const Bias bias)
  {
#ifdef RYUJIN_DEVICE_COMPILATION_PASS
    /* Call generic std::pow() implementation (in single precision) */
    return std::pow(static_cast<float>(x), static_cast<float>(b));
#elif DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)
    /* Use a custom fast_pow implementation instead of std::pow(): */
    return fast_pow(dealii::VectorizedArray<double, 2>(x), b, bias)[0];
#else
    /* Call generic std::pow() implementation (in single precision) */
    return std::pow(static_cast<float>(x), static_cast<float>(b));
#endif
  }

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
  DEAL_II_ALWAYS_INLINE inline T read_entry(const V &vector, unsigned int i)
  {
    static_assert(std::is_same_v<typename get_value_type<T>::type,
                                 typename V::value_type>,
                  "type mismatch");
    T result;

    if constexpr (std::is_same_v<T, typename get_value_type<T>::type>) {
      /* Non-vectorized sequential access. */
      result = vector.local_element(i);
    } else {
      /* Vectorized fast access. index must be divisible by simd_length */
      result.load(vector.get_values() + i);
    }

    return result;
  }


  /**
   * Variant of above function specialized for std::vector.
   * @ingroup SIMD
   */
  template <typename T, typename T2>
  DEAL_II_ALWAYS_INLINE inline T read_entry(const std::vector<T2> &vector,
                                            unsigned int i)
  {
    if constexpr (std::is_same_v<typename get_value_type<T>::type, T2>) {
      /* Optimized default for source and destination with same type: */

      T result;
      if constexpr (std::is_same_v<T, typename get_value_type<T>::type>) {
        /* Non-vectorized sequential access. */
        result = vector[i];
      } else {
        /* Vectorized fast access. index must be divisible by simd_length */
        result.load(vector.data() + i);
      }
      return result;

    } else {
      /* Fallback for mismatched types (float vs double): */
      T result;
      if constexpr (std::is_same_v<T, typename get_value_type<T>::type>) {
        result = vector[i];
      } else {
        // FIXME: suboptimal
        for (unsigned int k = 0; k < T::size(); ++k)
          result[k] = vector[i + k];
      }
      return result;
    }
  }


  /**
   * Return a VectorizedArray with
   *   { U[js[0] , U[js[1]] , ... , U[js[VectorizedArray::size() - 1]] }
   *
   * @ingroup SIMD
   */
  template <typename T, typename V>
  DEAL_II_ALWAYS_INLINE inline T read_entry(const V &vector,
                                            const unsigned int *js)
  {
    static_assert(std::is_same_v<typename get_value_type<T>::type,
                                 typename V::value_type>,
                  "type mismatch");
    T result;

    if constexpr (std::is_same_v<T, typename get_value_type<T>::type>) {
      /* Non-vectorized sequential access. */
      result = vector.local_element(js[0]);
    } else {
      /* Vectorized fast access. index must be divisible by simd_length */
      result.gather(vector.get_values(), js);
    }

    return result;
  }


  /**
   * Variant of above function specialized for std::vector.
   * @ingroup SIMD
   */
  template <typename T, typename T2>
  DEAL_II_ALWAYS_INLINE inline T read_entry(const std::vector<T2> &vector,
                                            const unsigned int *js)
  {
    static_assert(std::is_same_v<typename get_value_type<T>::type, T2>,
                  "type mismatch");
    T result;

    if constexpr (std::is_same_v<T, typename get_value_type<T>::type>) {
      /* Non-vectorized sequential access. */
      result = vector[js[0]];
    } else {
      /* Vectorized fast access. index must be divisible by simd_length */
      result.load(vector.data(), js);
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
  write_entry(V &vector, const T &values, unsigned int i)
  {
    static_assert(std::is_same_v<typename get_value_type<T>::type,
                                 typename V::value_type>,
                  "type mismatch");

    if constexpr (std::is_same_v<T, typename get_value_type<T>::type>) {
      /* Non-vectorized sequential access. */
      vector.local_element(i) = values;
    } else {
      /* Vectorized fast access. index must be divisible by simd_length */
      values.store(vector.get_values() + i);
    }
  }


  /**
   * Variant of above function specialized for std::vector.
   * @ingroup SIMD
   */
  template <typename T, typename T2>
  DEAL_II_ALWAYS_INLINE inline void
  write_entry(std::vector<T2> &vector, const T &values, unsigned int i)
  {
    if constexpr (std::is_same_v<typename get_value_type<T>::type, T2>) {
      /* Optimized default for source and destination with same type: */

      if constexpr (std::is_same_v<T, typename get_value_type<T>::type>) {
        /* Non-vectorized sequential access. */
        vector[i] = values;
      } else {
        /* Vectorized fast access. index must be divisible by simd_length */
        values.store(vector.data() + i);
      }

    } else {
      /* Fallback for mismatched types (float vs double): */
      if constexpr (std::is_same_v<T, typename get_value_type<T>::type>) {
        vector[i] = values;
      } else {
        // FIXME: suboptimal
        for (unsigned int k = 0; k < T::size(); ++k)
          vector[i + k] = values[k];
      }
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
                   const unsigned int k [[maybe_unused]])
  {
    Assert(k == 0,
           dealii::ExcMessage(
               "The given index k must be zero for a serial tensor"));
    return serial;
  }


  /**
   * Update the k-th serial component of a Tensor of VectorizedArray
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
                       const unsigned int k [[maybe_unused]])
  {
    Assert(k == 0,
           dealii::ExcMessage(
               "The given index k must be zero for a serial tensor"));

    result = serial;
  }

  //@}

} // namespace ryujin
