//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef HELPER_H
#define HELPER_H

#include "problem_description.h"

#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <type_traits>


namespace grendel
{
  /**
   * @name Packed index handling
   */
  //@{

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

  //@}
  /**
   * @name Serial access to arrays of vectors
   */
  //@{

  /**
   * Return a tensor populated with the entries
   *   { U[0][i] , U[1][i] , ... , U[k][i] }
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


  /**
   * Variant of above function returning a std::array instead.
   */
  template <typename T1, std::size_t k, typename T2>
  DEAL_II_ALWAYS_INLINE inline std::array<typename T1::value_type, k>
  gather_array(const std::array<T1, k> &U, const T2 i)
  {
    std::array<typename T1::value_type, k> result;
    for (unsigned int j = 0; j < k; ++j)
      result[j] = U[j].local_element(i);
    return result;
  }


  /**
   * Write out the given tensor @p values into
   *   { U[0][i] , U[1][i] , ... , U[k][i] }
   */
  template <typename T1, std::size_t k1, typename T2, typename T3>
  DEAL_II_ALWAYS_INLINE inline void
  scatter(std::array<T1, k1> &U, const T2 &values, const T3 i)
  {
    for (unsigned int j = 0; j < k1; ++j)
      U[j].local_element(i) = values[j];
  }

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
   */
  template <typename T1>
  DEAL_II_ALWAYS_INLINE inline void
  simd_store(T1 &vector,
             const dealii::VectorizedArray<typename T1::value_type> &values,
             unsigned int i)
  {
    values.store(access_data(vector) + i);
  }


  /**
   * Return a tensor of VectorizedArray entries. If the @p index_argument
   * is a plain `unsigned int` the result is
   *
   *   { U[0][i] , U[0][i+1] , ... , U[0][i+VectorizedArray::size()-1] }
   *   ...
   *   { U[k-1][i] , U[k-1][i+1] , ... , U[k-1][i+VectorizedArray::size()-1] }
   *
   * If the @p index_argument parameter is a pointer the result is
   *
   *   { U[0][js[0]] , U[0][js[0]] , ... , U[0][js[VectorizedArray::size()-1]] }
   *   ...
   *   { U[k-1][js[0]] , U[k-1][js[0]] , ... ,
   * U[k-1][js[VectorizedArray::size()-1]] }
   */
  template <typename T1, std::size_t k, typename T2>
  DEAL_II_ALWAYS_INLINE inline dealii::
      Tensor<1, k, dealii::VectorizedArray<typename T1::value_type>>
      simd_gather(const std::array<T1, k> &U, T2 index_argument)
  {
    dealii::Tensor<1, k, dealii::VectorizedArray<typename T1::value_type>>
        result;

    for (unsigned int j = 0; j < k; ++j)
      result[j] = simd_load(U[j], index_argument);

    return result;
  }


  /**
   * Variant of above function that returns an array instead of a tensor
   */
  template <typename T1, std::size_t k, typename T2>
  DEAL_II_ALWAYS_INLINE inline std::
      array<dealii::VectorizedArray<typename T1::value_type>, k>
      simd_gather_array(const std::array<T1, k> &U, const T2 index_argument)
  {
    std::array<dealii::VectorizedArray<typename T1::value_type>, k> result;

    for (unsigned int j = 0; j < k; ++j)
      result[j] = simd_load(U[j], index_argument);

    return result;
  }


  /**
   * Converse operation to simd_gather() and simd_gather_array()
   */
  template <typename T1, typename T2, typename T3>
  DEAL_II_ALWAYS_INLINE inline void
  simd_scatter(T1 &U, const T2 &result, T3 index_argument)
  {
    constexpr size_t k = std::tuple_size<T1>::value;
    for (unsigned int j = 0; j < k; ++j)
      simd_store(U[j], result[j], index_argument);
  }

  //@}
  /**
   * @name SIMD and serial vectorized transpose and store / load and
   * transpose operations:
   */
  //@{

  /**
   * FIXME: Write documentation
   */
  template <int dim, typename T1>
  DEAL_II_ALWAYS_INLINE inline auto simd_load_vlat(T1 &vector, unsigned int i)
  {
    using Number = typename T1::value_type;
    using PD = ProblemDescription<dim, dealii::VectorizedArray<Number>>;

    constexpr auto simd_length = dealii::VectorizedArray<Number>::size();
    constexpr unsigned int problem_dimension = PD::rank2_type::dimension;
    constexpr unsigned int flux_and_u_width = (dim + 1) * problem_dimension;

    const auto indices = generate_iterators<simd_length>(
        [&](auto k) -> unsigned int { return (i + k) * flux_and_u_width; });

    std::pair<typename PD::rank1_type, typename PD::rank2_type> result;

    vectorized_load_and_transpose(
        flux_and_u_width, vector.data(), indices.data(), &result.first[0]);

    return result;
  }


  /**
   * FIXME: Write documentation
   */
  template <int dim, typename T1>
  DEAL_II_ALWAYS_INLINE inline auto simd_load_vlat(T1 &vector,
                                                   const unsigned int *js)
  {
    using Number = typename T1::value_type;
    using PD = ProblemDescription<dim, dealii::VectorizedArray<Number>>;

    constexpr auto simd_length = dealii::VectorizedArray<Number>::size();
    constexpr unsigned int problem_dimension = PD::rank2_type::dimension;
    constexpr unsigned int flux_and_u_width = (dim + 1) * problem_dimension;

    const auto indices = generate_iterators<simd_length>(
        [&](auto k) -> unsigned int { return js[k] * flux_and_u_width; });

    std::pair<typename PD::rank1_type, typename PD::rank2_type> result;

    vectorized_load_and_transpose(
        flux_and_u_width, vector.data(), indices.data(), &result.first[0]);

    return result;
  }


  /**
   * FIXME: Write documentation
   */
  template <typename T1, typename T2>
  DEAL_II_ALWAYS_INLINE inline void
  simd_store_vtas(T1 &vector, const T2 &entries, unsigned int i)
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
   * Non-SIMD counterpart of vectorized load and transpose Read in the
   * object @p entries from the @p i th position of the @p vector object.
   */
  template <int dim, typename T1>
  DEAL_II_ALWAYS_INLINE inline auto load_vlat(const T1 &vector, unsigned int i)
  {
    using Number = typename T1::value_type;
    using PD = ProblemDescription<dim, Number>;

    constexpr unsigned int problem_dimension = PD::rank2_type::dimension;
    constexpr unsigned int flux_and_u_width = (dim + 1) * problem_dimension;

    typename PD::rank1_type U_i;
    typename PD::rank2_type f_i;

    for (unsigned int d = 0; d < problem_dimension; ++d)
      U_i[d] = vector[i * flux_and_u_width + d];
    for (unsigned int d = 0; d < problem_dimension; ++d)
      for (unsigned int e = 0; e < dim; ++e)
        f_i[d][e] =
            vector[i * flux_and_u_width + problem_dimension + d * dim + e];

    return std::make_pair(U_i, f_i);
  }


  /**
   * Non-SIMD counterpart of vectorized transpose and store: Write out the
   * object @p entries to the @p i th position of the @p vector
   * object.
   */
  template <typename T1, typename T2>
  DEAL_II_ALWAYS_INLINE inline void
  store_vtas(T1 &vector, const T2 &entries, unsigned int i)
  {
    using Number = typename T1::value_type;
    constexpr unsigned int flux_and_u_width = sizeof(entries) / sizeof(Number);

    const Number *values = &entries.first[0];

    for (unsigned int k = 0; k < flux_and_u_width; ++k)
      vector[i * flux_and_u_width + k] = values[k];
  }

  //@}


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
