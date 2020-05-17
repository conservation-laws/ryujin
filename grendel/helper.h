//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef HELPER_H
#define HELPER_H

#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <type_traits>

namespace dealii
{
  namespace LinearAlgebra
  {
    namespace distributed
    {
      template <typename T, typename MemorySpace>
      class Vector;
    }
  } // namespace LinearAlgebra
} // namespace dealii


namespace grendel
{

  /*
   * Packed iterator handling.
   */

  namespace
  {
    template <typename Functor, size_t... Is>
    auto generate_iterators_impl(Functor f, std::index_sequence<Is...>)
        -> std::array<decltype(f(0)), sizeof...(Is)>
    {
      return {f(Is)...};
    }
  } /* namespace */

  /**
   * Given a callable object f(k), this function creates a std::array with
   * elements initialized as follows:
   *
   *   { f(0) , f(1) , ... , f(length - 1) }
   *
   * We use this function to create an array of sparsity iterators that
   * cannot be default initialized.
   */
  template <unsigned int length, typename Functor>
  DEAL_II_ALWAYS_INLINE inline auto generate_iterators(Functor f)
      -> std::array<decltype(f(0)), length>
  {
    return generate_iterators_impl<>(f, std::make_index_sequence<length>());
  }


  /**
   * Increment all iterators in an std::array simultaneously.
   */
  template <typename T>
  DEAL_II_ALWAYS_INLINE inline void increment_iterators(T &iterators)
  {
    for (auto &it : iterators)
      it++;
  }


  /*
   * Serial access to arrays of vectors
   */


  template <typename T1, std::size_t k, typename T2>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, k, typename T1::value_type>
  gather(const std::array<T1, k> &U, const T2 i)
  {
    dealii::Tensor<1, k, typename T1::value_type> result;
    for (unsigned int j = 0; j < k; ++j)
      result[j] = U[j].local_element(i);
    return result;
  }


  template <typename T1, std::size_t k, typename T2>
  DEAL_II_ALWAYS_INLINE inline std::array<typename T1::value_type, k>
  gather_array(const std::array<T1, k> &U, const T2 i)
  {
    std::array<typename T1::value_type, k> result;
    for (unsigned int j = 0; j < k; ++j)
      result[j] = U[j].local_element(i);
    return result;
  }


  template <typename T1, std::size_t k1, typename T2, typename T3>
  DEAL_II_ALWAYS_INLINE inline void
  scatter(std::array<T1, k1> &U, const T2 &result, const T3 i)
  {
    for (unsigned int j = 0; j < k1; ++j)
      U[j].local_element(i) = result[j];
  }


  /*
   * SIMD based access to vectors and arrays of vectors
   */

  /**
   * Populate a VectorizedArray with
   *   { U[i] , U[i + 1] , ... , U[i + VectorizedArray::size() - 1] }
   */
  template <typename T1>
  DEAL_II_ALWAYS_INLINE inline dealii::VectorizedArray<typename T1::value_type>
  simd_gather(const T1 &U, unsigned int i)
  {
    dealii::VectorizedArray<typename T1::value_type> result;
    result.load(U.get_values() + i);
    return result;
  }


  /**
   * Populate an array of VectorizedArray with
   *   { U[0][i] , U[0][i+1] , ... , U[0][i+VectorizedArray::size()-1] }
   *   ...
   *   { U[k-1][i] , U[k-1][i+1] , ... , U[k-1][i+VectorizedArray::size()-1] }
   */
  template <typename T1, std::size_t k>
  DEAL_II_ALWAYS_INLINE inline dealii::
      Tensor<1, k, dealii::VectorizedArray<typename T1::value_type>>
      simd_gather(const std::array<T1, k> &U, unsigned int i)
  {
    dealii::Tensor<1, k, dealii::VectorizedArray<typename T1::value_type>>
        result;

    for (unsigned int j = 0; j < k; ++j)
      result[j].load(U[j].get_values() + i);

    return result;
  }


  /**
   * Variant of above function that returns an array instead of a tensor
   */
  template <typename T1, std::size_t k, typename T2>
  DEAL_II_ALWAYS_INLINE inline std::
      array<dealii::VectorizedArray<typename T1::value_type>, k>
      simd_gather_array(const std::array<T1, k> &U, const T2 i)
  {
    std::array<dealii::VectorizedArray<typename T1::value_type>, k> result;

    for (unsigned int j = 0; j < k; ++j)
      result[j].load(U[j].get_values() + i);

    return result;
  }


  template <typename T1>
  DEAL_II_ALWAYS_INLINE inline dealii::VectorizedArray<typename T1::value_type>
  simd_gather(
      const T1 &U,
      const std::array<
          unsigned int,
          dealii::VectorizedArray<typename T1::value_type>::size()>
          js)
  {
    dealii::VectorizedArray<typename T1::value_type> result;
    result.gather(U.get_values(), js.data());
    return result;
  }


  template <typename T1>
  DEAL_II_ALWAYS_INLINE inline dealii::VectorizedArray<typename T1::value_type>
  simd_gather(const T1 &U, const unsigned int *js)
  {
    dealii::VectorizedArray<typename T1::value_type> result;
    result.gather(U.get_values(), js);
    return result;
  }


  /*
   * It's magic
   */
  template <typename T1, std::size_t k>
  DEAL_II_ALWAYS_INLINE inline dealii::
      Tensor<1, k, dealii::VectorizedArray<typename T1::value_type>>
      simd_gather(
          const std::array<T1, k> &U,
          const std::array<unsigned int,
                           dealii::VectorizedArray<
                               typename T1::value_type>::size()> js)
  {
    dealii::Tensor<1, k, dealii::VectorizedArray<typename T1::value_type>>
        result;

    for (unsigned int j = 0; j < k; ++j)
      result[j].gather(U[j].get_values(), js.data());

    return result;
  }

  template <typename T1, std::size_t k>
  DEAL_II_ALWAYS_INLINE inline dealii::
      Tensor<1, k, dealii::VectorizedArray<typename T1::value_type>>
      simd_gather(const std::array<T1, k> &U, const unsigned int *js)
  {
    dealii::Tensor<1, k, dealii::VectorizedArray<typename T1::value_type>>
        result;

    for (unsigned int j = 0; j < k; ++j)
      result[j].gather(U[j].get_values(), js);

    return result;
  }


  /**
   * FIXME: Write documentation
   */
  template <typename T, typename M>
  DEAL_II_ALWAYS_INLINE inline void
  simd_scatter(dealii::LinearAlgebra::distributed::Vector<T, M> &vector,
               const dealii::VectorizedArray<T> &values,
               unsigned int i)
  {
    values.store(vector.get_values() + i);
  }


  /**
   * FIXME: Write documentation
   */
  template <typename T>
  DEAL_II_ALWAYS_INLINE inline void
  simd_scatter(dealii::AlignedVector<T> &vector,
               const dealii::VectorizedArray<T> &values,
               unsigned int i)
  {
    values.store(vector.data() + i);
  }


  /**
   * FIXME: Write documentation
   */
  template <typename T1, typename T2>
  DEAL_II_ALWAYS_INLINE inline void
  simd_scatter(T1 &U, const T2 &result, unsigned int i)
  {
    constexpr size_t k = std::tuple_size<T1>::value;
    for (unsigned int j = 0; j < k; ++j)
      simd_scatter(U[j], result[j], i);
  }


  /**
   * FIXME: Write documentation
   */
  template <typename T1, typename T2>
  DEAL_II_ALWAYS_INLINE inline void
  simd_scatter_vtas(T1 &vector, const T2 &entries, unsigned int i)
  {
    using Number = typename T1::value_type;

    constexpr auto simd_length = dealii::VectorizedArray<Number>::size();
    constexpr unsigned int flux_and_u_width =
        sizeof(entries) / sizeof(Number) / simd_length;

    const auto indices = generate_iterators<simd_length>(
        [&](auto k) -> unsigned int { return (i + k) * flux_and_u_width; });

    vectorized_transpose_and_store(false,
                                   flux_and_u_width,
                                   &entries.first[0],
                                   indices.data(),
                                   vector.data());
  }


  /**
   * FIXME: Write documentation
   */
  template <typename T1, typename T2>
  DEAL_II_ALWAYS_INLINE inline void
  scatter_vtas(T1 &vector, const T2 &entries, unsigned int i)
  {
    using Number = typename T1::value_type;
    constexpr unsigned int flux_and_u_width = sizeof(entries) / sizeof(Number);

    const Number *values = &entries.first[0];

    for (unsigned int k = 0; k < flux_and_u_width; ++k)
      vector[i * flux_and_u_width + k] = values[k];
  }


#ifndef DOXYGEN
  namespace
  {
    template <int dim, typename Number, typename Callable>
    class ToFunction : public dealii::Function<dim, Number>
    {
    public:
      ToFunction(const Callable &callable, const unsigned int k)
          : dealii::Function<dim, Number>(1)
          , callable_(callable)
          , k_(k)
      {
      }

      virtual Number value(const dealii::Point<dim> &point,
                           unsigned int /*component*/) const
      {
        return callable_(point)[k_];
      }

    private:
      const Callable callable_;
      const unsigned int k_;
    };
  } // namespace
#endif


  /**
   * Convenience wrapper that creates a dealii::Function object out of a
   * (fairly general) callable object:
   */
  template <int dim, typename Number, typename Callable>
  ToFunction<dim, Number, Callable> to_function(const Callable &callable,
                                                const unsigned int k)
  {
    return {callable, k};
  }


} // namespace grendel


/*
 * AssertThrowSIMD
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


/*
 * A convenience macro that automatically writes out an accessor (or
 * getter) function:
 *
 *   const Foo& bar() const
 *   {
 *      return bar_;
 *   }
 *
 * or
 *
 *   const Foo& bar() const
 *   {
 *      return *bar_;
 *   }
 *
 * depending on whether bar_ can be dereferenced, or not.
 */

namespace
{
  template <typename T>
  class is_dereferenciable
  {
    template <typename C>
    static auto test(...) -> std::false_type;

    template <typename C>
    static auto test(C *) -> decltype(*std::declval<C>(), std::true_type());

  public:
    typedef decltype(test<T>(nullptr)) type;
    static constexpr auto value = type::value;
  };

  template <typename T, typename>
  auto dereference(T &t) -> decltype(dereference(*t)) &;

  template <
      typename T,
      typename = typename std::enable_if<!is_dereferenciable<T>::value>::type>
  const T &dereference(T &t)
  {
    return t;
  }

  template <
      typename T,
      typename = typename std::enable_if<is_dereferenciable<T>::value>::type>
  auto dereference(T &t) -> const decltype(*t) &
  {
    return *t;
  }
} /* anonymous namespace */

#define ACCESSOR_READ_ONLY(member)                                             \
public:                                                                        \
  decltype(dereference(member##_)) &member() const                             \
  {                                                                            \
    return dereference(member##_);                                             \
  }                                                                            \
                                                                               \
protected:

#define ACCESSOR_READ_ONLY_NO_DEREFERENCE(member)                              \
public:                                                                        \
  const decltype(member##_) &member() const                                    \
  {                                                                            \
    return member##_;                                                          \
  }                                                                            \
                                                                               \
protected:


/*
 * OpenMP parallel for loop options and macros:
 */

#define GRENDEL_PRAGMA(x) _Pragma(#x)

#define GRENDEL_PARALLEL_REGION_BEGIN                                          \
  GRENDEL_PRAGMA(omp parallel default(shared))                                 \
  {

#define GRENDEL_PARALLEL_REGION_END }

#define GRENDEL_OMP_FOR GRENDEL_PRAGMA(omp for)
#define GRENDEL_OMP_FOR_NOWAIT GRENDEL_PRAGMA(omp for nowait)
#define GRENDEL_OMP_BARRIER GRENDEL_PRAGMA(omp barrier)

#define GRENDEL_LIKELY(x) (__builtin_expect(!!(x), 1))
#define GRENDEL_UNLIKELY(x) (__builtin_expect(!!(x), 0))

#endif /* HELPER_H */
