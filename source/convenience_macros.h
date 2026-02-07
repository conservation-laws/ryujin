//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>

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

      Number value(const dealii::Point<dim> &point,
                   unsigned int /*component*/) const override
      {
        return callable_(point)[k_];
      }

    private:
      const Callable callable_;
      const unsigned int k_;
    };


    template <int dim, typename Number, typename Callable>
    class ToVectorFunction : public dealii::Function<dim, Number>
    {
    public:
      ToVectorFunction(const Callable &callable, const unsigned int components)
          : dealii::Function<dim, Number>(components)
          , callable_(callable)
      {
      }

      Number value(const dealii::Point<dim> &point,
                   unsigned int component) const override
      {
        return callable_(point)[component];
      }

      void vector_value(const dealii::Point<dim> &point,
                        dealii::Vector<double> &values) const override
      {
        AssertDimension(values.size(), this->n_components);

        const auto temp = callable_(point);
        for (unsigned int k = 0; k < this->n_components; ++k)
          values(k) = temp[k];
      }

    private:
      const Callable callable_;
    };
  } // namespace
#endif

  /**
   * @name Various convenience wrappers for dealing with dealii::Function,
   * dealii::Tensor:
   */
  //@{


  /**
   * Convenience wrapper that creates a (scalar) dealii::Function object
   * out of a (fairly general) callable object returning array-like values.
   * An example usage is given by the interpolation of initial values
   * performed in InitialValues::interpolate_hyperbolic_vector() and
   * InitialValues::interpolate_initial_precomputed_vector()
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
   * Convenience wrapper that creates a vector-valued dealii::Function
   * object out of a (fairly general) callable object returning array-like
   * values. An example usage is given by the interpolation of initial
   * values performed in InitialValues::interpolate_hyperbolic_vector() and
   * InitialValues::interpolate_initial_precomputed_vector()
   * ```
   * dealii::VectorTools::interpolate(
   *   dof_handler,
   *   to_function<dim, Number>(callable, block_size),
   *   block_vector);
   * ```
   *
   * @param callable A callable object that provides an `operator(const
   * Point<dim> &)` and returns an array or rank-1 tensor. More precisely,
   * the return type must have a subscript operator `operator[]`.
   *
   * @param n_components number of components.
   *
   * @ingroup Miscellaneous
   */
  template <int dim, typename Number, typename Callable>
  ToVectorFunction<dim, Number, Callable>
  to_vector_function(const Callable &callable, const unsigned int n_components)
  {
    return {callable, n_components};
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

  //@}
} // namespace ryujin


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

  template <typename T>
  auto dereference(T &t) -> T &
    requires(!is_dereferenceable<T>::value)
  {
    return t;
  }

  template <typename T>
  auto dereference(T &t) -> decltype(*t) &
    requires is_dereferenceable<T>::value
  {
    return *t;
  }
} /* anonymous namespace */
#endif

/**
 * @name Macros for accessor definitions with automatic dereferencing
 */
//@{

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
  inline const auto &member() const                                            \
  {                                                                            \
    return dereference(member##_);                                             \
  }


/**
 * Variant of the macro above that returns a mutable reference.
 *
 * @ingroup Miscellaneous
 */
#define ACCESSOR(member)                                                       \
  inline auto &member()                                                        \
  {                                                                            \
    return dereference(member##_);                                             \
  }


/**
 * Variant of the macro above that does not attempt to dereference the
 * underlying object.
 *
 * @ingroup Miscellaneous
 */
#define ACCESSOR_READ_ONLY_NO_DEREFERENCE(member)                              \
  inline const auto &member() const                                            \
  {                                                                            \
    return member##_;                                                          \
  }


/**
 * Variant of the macro above that takes two arguments, container and
 * member, and creates an accessor function.
 * ```
 * const Foo& member() const { return container_.member; }
 * ```
 * The function will dereference container and member if they are of
 * pointer type.
 *
 * @ingroup Miscellaneous
 */
#define ACCESSOR_CONTAINER_READ_ONLY(container, member)                        \
  inline const auto &member() const                                            \
  {                                                                            \
    return dereference(dereference(container).member);                         \
  }


//@}
/**
 * @name Macros for compiler hints
 */
//@{

/**
 * Macro expanding to a `#pragma` directive that looks nicer in indented
 * code and can be used in other preprocessor macro definitions.
 *
 * @ingroup Miscellaneous
 */
#define RYUJIN_PRAGMA(x) _Pragma(#x)


/**
 * Compiler hint annotating a boolean to be likely true.
 *
 * Intended use:
 * ```
 * if (RYUJIN_LIKELY(thread_ready == true)) {
 *   // likely branch
 * }
 * ```
 *
 * @note The performance penalty of incorrectly marking a condition as
 * likely is severe. Use only if the condition is almost always true.
 * @ingroup Miscellaneous
 */
#define RYUJIN_LIKELY(x) (__builtin_expect(!!(x), 1))


/**
 * Compiler hint annotating a boolean expression to be likely false.
 *
 * Intended use:
 * ```
 * if (RYUJIN_UNLIKELY(thread_ready == false)) {
 *   // unlikely branch
 * }
 * ```
 *
 * @note The performance penalty of incorrectly marking a condition as
 * unlikely is severe. Use only if the condition is almost always false.
 * @ingroup Miscellaneous
 */
#define RYUJIN_UNLIKELY(x) (__builtin_expect(!!(x), 0))


/**
 * Injects a label into the generated assembly.
 *
 * @ingroup Miscellaneous
 */
#define ASM_LABEL(label) asm("#" label);

//@}
