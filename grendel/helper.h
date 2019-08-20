#ifndef HELPER_H
#define HELPER_H

#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <type_traits>


namespace std
{
  //
  // FIXME: Refactor into library
  //

  template <typename Number, int width, typename Number2>
  inline ::dealii::VectorizedArray<Number, width>
  pow(const ::dealii::VectorizedArray<Number, width> &x, const Number2 p)
  {
    Number values[::dealii::VectorizedArray<Number, width>::n_array_elements];
    for (unsigned int i = 0;
         i < dealii::VectorizedArray<Number, width>::n_array_elements;
         ++i)
      values[i] = std::pow(x[i], p);
    ::dealii::VectorizedArray<Number, width> out;
    out.load(&values[0]);
    return out;
  }
}


namespace dealii
{
  //
  // FIXME: Refactor into library and document
  //

  DEAL_II_ALWAYS_INLINE inline VectorizedArray<float>
  ternary_gt(const VectorizedArray<float> &left,
             const VectorizedArray<float> &right,
             const VectorizedArray<float> &true_value,
             const VectorizedArray<float> &false_value)
  {
    const auto mask = _mm256_cmp_ps(left.data, right.data, _CMP_GT_OQ);
    VectorizedArray<float> result;
    result.data = _mm256_or_ps(_mm256_and_ps(mask, true_value.data),
                               _mm256_andnot_ps(mask, false_value.data));
    return result;
  }

  DEAL_II_ALWAYS_INLINE inline VectorizedArray<double>
  ternary_gt(const VectorizedArray<double> &left,
             const VectorizedArray<double> &right,
             const VectorizedArray<double> &true_value,
             const VectorizedArray<double> &false_value)
  {
    const auto mask = _mm256_cmp_pd(left.data, right.data, _CMP_GT_OQ);
    VectorizedArray<double> result;
    result.data = _mm256_or_pd(_mm256_and_pd(mask, true_value.data),
                               _mm256_andnot_pd(mask, false_value.data));
    return result;
  }


  DEAL_II_ALWAYS_INLINE inline double ternary_gt(const double left,
                                                 const double right,
                                                 const double true_value,
                                                 const double false_value)
  {
    return (left > right) ? true_value : false_value;
  }


  DEAL_II_ALWAYS_INLINE inline float ternary_gt(const float left,
                                                const float right,
                                                const float true_value,
                                                const float false_value)
  {
    return (left > right) ? true_value : false_value;
  }

} // namespace dealii



namespace grendel
{
  /*
   * It's magic.
   *
   * There is no native functionality to access a matrix entry by providing
   * an iterator over the sparsity pattern. This is silly: The sparsity
   * pattern iterator alreay *knows* the exact location in the matrix
   * vector. Thus, this little workaround.
   */
  template <typename Matrix, typename Iterator>
  DEAL_II_ALWAYS_INLINE inline typename Matrix::value_type
  get_entry(const Matrix &matrix, const Iterator &it)
  {
    const auto global_index = it->global_index();
    const typename Matrix::const_iterator matrix_iterator(&matrix,
                                                          global_index);
    return matrix_iterator->value();
  }


  /*
   * It's magic
   */
  template <typename Matrix, typename Iterator>
  DEAL_II_ALWAYS_INLINE inline void set_entry(Matrix &matrix,
                                              const Iterator &it,
                                              typename Matrix::value_type value)
  {
    const auto global_index = it->global_index();
    typename Matrix::iterator matrix_iterator(&matrix, global_index);
    matrix_iterator->value() = value;
  }


  /*
   * It's magic
   */
  template <typename T1, std::size_t k, typename T2>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, k>
  gather_get_entry(const std::array<T1, k> &U, const T2 it)
  {
    dealii::Tensor<1, k> result;
    for (unsigned int j = 0; j < k; ++j)
      result[j] = get_entry(U[j], it);
    return result;
  }


  /*
   * It's magic
   *
   * FIXME: k versus l
   */
  template <typename T1, std::size_t k, int l, typename T2>
  DEAL_II_ALWAYS_INLINE inline void scatter_set_entry(
      std::array<T1, k> &U, const T2 it, const dealii::Tensor<1, l> &V)
  {
    for (unsigned int j = 0; j < k; ++j)
      set_entry(U[j], it, V[j]);
  }


  /*
   * It's magic
   */
  template <typename T1, std::size_t k, typename T2, typename T3>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, k>
  gather(const std::array<T1, k> &U, const T2 i, const T3 l)
  {
    dealii::Tensor<1, k> result;
    for (unsigned int j = 0; j < k; ++j)
      result[j] = U[j](i, l);
    return result;
  }


  /*
   * It's magic
   */
  template <typename T1, std::size_t k, typename T2>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, k>
  gather(const std::array<T1, k> &U, const T2 i)
  {
    dealii::Tensor<1, k> result;
    for (unsigned int j = 0; j < k; ++j)
      result[j] = U[j].local_element(i);
    return result;
  }


  /*
   * It's magic
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


  /*
   * It's magic
   */

  template <typename T1, std::size_t k1, typename T2, typename T3>
  DEAL_II_ALWAYS_INLINE inline void
  scatter(std::array<T1, k1> &U, const T2 &result, const T3 i)
  {
    for (unsigned int j = 0; j < k1; ++j)
      U[j].local_element(i) = result[j];
  }


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


  /*
   * It's magic
   */
  template <int dim, typename Number, typename Callable>
  ToFunction<dim, Number, Callable> to_function(const Callable &callable,
                                                const unsigned int k)
  {
    return {callable, k};
  }


  // FIXME: Refactor - do we have something like this in the library?
  template <typename T>
  struct get_value_type {
    using type = T;
  };

  template <typename T>
  struct get_value_type<dealii::VectorizedArray<T>> {
    using type = T;
  };

} // namespace grendel


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



#endif /* HELPER_H */
