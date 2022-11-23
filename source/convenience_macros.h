//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <deal.II/base/function.h>

/**
 * @name Various convenience functions and macros
 */
//@{

namespace ryujin
{
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
   * Convenience wrapper that creates a (scalar) dealii::Function object
   * out of a (fairly general) callable object returning array-like values.
   * An example usage is given by the interpolation of initial values
   * performed in InitialValues::interpolate();
   * ```
   * for(unsigned int i = 0; i < problem_dimension; ++i)
   *   dealii::VectorTools::interpolate(
   *     dof_handler,
   *     to_function<dim, Number>(callable, i),
   *     U[i]);
   * ```
   *
   * @param callable A callable object that provides an `operator(const
   * Point<dim> &)` and returns an array or rank-1 tensor. More precisely,
   * the return type must have a subscript operator `operator[]`.
   *
   * @param k Index describing the component that is returned by the
   * function object.
   *
   * @ingroup Miscellaneous
   */
  template <int dim, typename Number, typename Callable>
  ToFunction<dim, Number, Callable> to_function(const Callable &callable,
                                                const unsigned int k)
  {
    return {callable, k};
  }


  /**
   * Contract a given rank-2 tensor flux_ij and a rank-1 tensor c_ij:
   */
  template <typename FT,
            int problem_dim = FT::dimension,
            typename TT = typename FT::value_type,
            typename T = typename TT::value_type>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, T>
  contract(const FT &flux_ij, const TT &c_ij)
  {
    dealii::Tensor<1, problem_dim, T> result;
    for (unsigned int k = 0; k < problem_dim; ++k)
      result[k] = flux_ij[k] * c_ij;
    return result;
  }


  /**
   * Add two given rank-2 tensors flux_left_ij and flux_right_ij:
   */
  template <typename FT, int problem_dim = FT::dimension>
  DEAL_II_ALWAYS_INLINE inline FT add(const FT &flux_left_ij,
                                      const FT &flux_right_ij)
  {
    FT result;
    for (unsigned int k = 0; k < problem_dim; ++k)
      result[k] = flux_left_ij[k] + flux_right_ij[k];
    return result;
  }


} // namespace ryujin


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

#ifndef DOXYGEN
namespace
{
  template <typename T>
  class is_dereferenceable
  {
    template <typename C>
    static auto test(...) -> std::false_type;

    template <typename C>
    static auto test(C *) -> decltype(*std::declval<C>(), std::true_type());

  public:
    using type = decltype(test<T>(nullptr));
    static constexpr auto value = type::value;
  };

  template <typename T, typename>
  auto dereference(T &t) -> decltype(dereference(*t)) &;

  template <
      typename T,
      typename = typename std::enable_if<!is_dereferenceable<T>::value>::type>
  const T &dereference(T &t)
  {
    return t;
  }

  template <
      typename T,
      typename = typename std::enable_if<is_dereferenceable<T>::value>::type>
  auto dereference(T &t) -> const decltype(*t) &
  {
    return *t;
  }
} /* anonymous namespace */
#endif

/**
 * A convenience macro that automatically writes out an accessor (or
 * getter) function:
 * ```
 * const Foo& bar() const { return bar_; }
 * ```
 * or
 * ```
 * const Foo& bar() const { return *bar_; }
 * ```
 * depending on whether bar_ can be dereferenced, or not.
 *
 * @ingroup Miscellaneous
 */
#define ACCESSOR_READ_ONLY(member)                                             \
public:                                                                        \
  DEAL_II_ALWAYS_INLINE inline decltype(dereference(member##_)) &member()      \
      const                                                                    \
  {                                                                            \
    return dereference(member##_);                                             \
  }                                                                            \
                                                                               \
protected:

/**
 * Variant of the macro above that does not attempt to dereference the
 * underlying object.
 *
 * @ingroup Miscellaneous
 */
#define ACCESSOR_READ_ONLY_NO_DEREFERENCE(member)                              \
public:                                                                        \
  DEAL_II_ALWAYS_INLINE inline const decltype(member##_) &member() const       \
  {                                                                            \
    return member##_;                                                          \
  }                                                                            \
                                                                               \
protected:


/**
 * Injects a label into the generated assembly.
 *
 * @ingroup Miscellaneous
 */
#define ASM_LABEL(label) asm("#" label);

//@}
