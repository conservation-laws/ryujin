#ifndef HELPER_H
#define HELPER_H

#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <type_traits>


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
   *   { f(0) , f(1) , ... , f(n_array_elements - 1) }
   *
   * We use this function to create an array of sparsity iterators that
   * cannot be default initialized.
   */
  template <unsigned int n_array_elements, typename Functor>
  DEAL_II_ALWAYS_INLINE inline auto generate_iterators(Functor f)
      -> std::array<decltype(f(0)), n_array_elements>
  {
    return generate_iterators_impl<>(
        f, std::make_index_sequence<n_array_elements>());
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


  /**
   * Convenience function that transforms an array of SparsityPattern
   * iterators into an array of the corresponding column indices
   */
  template <typename T, std::size_t n_array_elements>
  DEAL_II_ALWAYS_INLINE inline std::array<unsigned int, n_array_elements>
  get_column_indices(const std::array<T, n_array_elements> &jts)
  {
    std::array<unsigned int, n_array_elements> result;
    for (unsigned int k = 0; k < n_array_elements; ++k)
      result[k] = jts[k]->column();
    return result;
  }


  /**
   * Convenience function that transforms an array of SparsityPattern
   * iterators into an array of the corresponding column indices
   */
  template <typename T, std::size_t n_array_elements>
  DEAL_II_ALWAYS_INLINE inline std::array<unsigned int, n_array_elements>
  get_global_indices(const std::array<T, n_array_elements> &jts)
  {
    std::array<unsigned int, n_array_elements> result;
    for (unsigned int k = 0; k < n_array_elements; ++k)
      result[k] = jts[k]->global_index();
    return result;
  }


  /*
   * Serial and SIMD iterator-based access to matrix values.
   */


  /*
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


  /**
   * SIMD variant of above function
   */
  template <typename Matrix, typename Iterator>
  DEAL_II_ALWAYS_INLINE inline dealii::VectorizedArray<
      typename Matrix::value_type>
  get_entry(
      const Matrix &matrix,
      const std::array<Iterator,
                       dealii::VectorizedArray<
                           typename Matrix::value_type>::n_array_elements> &its)
  {
    dealii::VectorizedArray<typename Matrix::value_type> result;

    // FIXME: This const_cast is terrible. Unfortunately, there is
    // currently no other way of extracting a raw pointer to the underlying
    // data vector of a SparseMatrix.
    const typename Matrix::value_type &data =
        const_cast<Matrix *>(&matrix)->diag_element(0);

    const auto indices = get_global_indices(its);
    result.gather(&data, indices.data());

    return result;
  }

  /**
   * SIMD variant of get_entry that returns
   *   { matrix.diag_element(i) , ... , matrix.diag_element(i + n_array_elements
   * - 1) }
   *
   * FIXME: Performance
   */
  template <typename Matrix>
  DEAL_II_ALWAYS_INLINE inline dealii::VectorizedArray<
      typename Matrix::value_type>
  simd_get_diag_element(const Matrix &matrix, unsigned int i)
  {
    dealii::VectorizedArray<typename Matrix::value_type> result;
    for (unsigned int k = 0;
         k <
         dealii::VectorizedArray<typename Matrix::value_type>::n_array_elements;
         ++k)
      result[k] = matrix.diag_element(i + k);
    return result;
  }


  /**
   * This is a vectorized variant of get_entry for the c_ij, and n_ij
   * "matrices" that are std::array<SparseMatrix<Number>, dim> objects.
   *
   * This provides both, the serial and SIMD variant.
   */
  template <typename T1, std::size_t k, typename T2>
  DEAL_II_ALWAYS_INLINE inline auto gather_get_entry(const std::array<T1, k> &U,
                                                     const T2 it)
      -> dealii::Tensor<1, k, decltype(get_entry(U[0], it))>
  {
    dealii::Tensor<1, k, decltype(get_entry(U[0], it))> result;
    for (unsigned int j = 0; j < k; ++j)
      result[j] = get_entry(U[j], it);
    return result;
  }


  /*
   * Variant of above function that takes a tuple of indices (i, l) instead
   * of an iterator it. This access is slow, avoid.
   *
   * FIXME: Refactor into something smarter...
   */
  template <typename T1, std::size_t k, typename T2, typename T3>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, k, typename T1::value_type>
  gather_get_entry(const std::array<T1, k> &U, const T2 i, const T3 l)
  {
    dealii::Tensor<1, k, typename T1::value_type> result;
    for (unsigned int j = 0; j < k; ++j)
      result[j] = U[j](i, l);
    return result;
  }


  /*
   * There is no native functionality to access a matrix entry by providing
   * an iterator over the sparsity pattern. This is silly: The sparsity
   * pattern iterator alreay *knows* the exact location in the matrix
   * vector. Thus, this little workaround.
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


  /**
   * Return the entry on the transposed position to the given
   * iterator, i.e., matrix(it->column(), it->row()). Read-only
   * version.
   */
  template <typename Matrix, typename Iterator>
  DEAL_II_ALWAYS_INLINE inline typename Matrix::value_type
  get_transposed_entry(const Matrix &matrix,
                       const Iterator &it,
                       const std::vector<unsigned int> &transposed_indices)
  {
    Iterator iterator_transpose =
        matrix.get_sparsity_pattern().begin(it->column()) +
        transposed_indices[it->global_index()];
    const auto global_index = iterator_transpose->global_index();
    typename Matrix::const_iterator matrix_iterator(&matrix, global_index);
    return matrix_iterator->value();
  }


  /**
   * Write the given value on the transposed position to the given
   * iterator, i.e., matrix(it->column(), it->row()) = value.
   */
  template <typename Matrix, typename Iterator>
  DEAL_II_ALWAYS_INLINE inline void
  set_transposed_entry(Matrix &matrix,
                       const Iterator &it,
                       const std::vector<unsigned int> &transposed_indices,
                       typename Matrix::value_type value)
  {
    Iterator iterator_transpose =
        matrix.get_sparsity_pattern().begin(it->column()) +
        transposed_indices[it->global_index()];
    const auto global_index = iterator_transpose->global_index();
    typename Matrix::iterator matrix_iterator(&matrix, global_index);
    matrix_iterator->value() = value;
  }


  /**
   * SIMD variant of above function
   */
  template <typename Matrix, typename Iterator>
  DEAL_II_ALWAYS_INLINE inline void set_entry(
      Matrix &matrix,
      const std::array<Iterator,
                       dealii::VectorizedArray<
                           typename Matrix::value_type>::n_array_elements> &its,
      const dealii::VectorizedArray<typename Matrix::value_type> &values)
  {
    typename Matrix::value_type &data = matrix.diag_element(0);
    const auto indices = get_global_indices(its);
    values.scatter(indices.data(), &data);
  }


  /**
   * This is a vectorized variant of get_entry for the c_ij, and n_ij
   * "matrices" that are std::array<SparseMatrix<Number>, dim> objects.
   *
   * This provides both, the serial and SIMD variant.
   *
   * FIXME: k versus l
   */
  template <typename T1, std::size_t k, int l, typename T2, typename T3>
  DEAL_II_ALWAYS_INLINE inline void scatter_set_entry(
      std::array<T1, k> &U, const T2 it, const dealii::Tensor<1, l, T3> &V)
  {
    for (unsigned int j = 0; j < k; ++j)
      set_entry(U[j], it, V[j]);
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
   *   { U[i] , U[i + 1] , ... , U[i + n_array_elements - 1] }
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
   *   { U[0][i] , U[0][i+1] , ... , U[0][i+n_array_elements-1] }
   *   ...
   *   { U[k-1][i] , U[k-1][i+1] , ... , U[k-1][i+n_array_elements-1] }
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


  /*
   * It's magic
   */
  template <typename T1>
  DEAL_II_ALWAYS_INLINE inline dealii::VectorizedArray<typename T1::value_type>
  simd_gather(
      const T1 &U,
      const std::array<
          unsigned int,
          dealii::VectorizedArray<typename T1::value_type>::n_array_elements>
          js)
  {
    dealii::VectorizedArray<typename T1::value_type> result;
    result.gather(U.get_values(), js.data());
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
                               typename T1::value_type>::n_array_elements> js)
  {
    dealii::Tensor<1, k, dealii::VectorizedArray<typename T1::value_type>>
        result;

    for (unsigned int j = 0; j < k; ++j)
      result[j].gather(U[j].get_values(), js.data());

    return result;
  }


  template <typename T1>
  DEAL_II_ALWAYS_INLINE inline void
  simd_scatter(T1 &U,
               const dealii::VectorizedArray<typename T1::value_type> &result,
               unsigned int i)
  {
    result.store(U.get_values() + i);
  }


  template <typename T1, std::size_t k, typename T2>
  DEAL_II_ALWAYS_INLINE inline void
  simd_scatter(std::array<T1, k> &U, const T2 &result, unsigned int i)
  {
    for (unsigned int j = 0; j < k; ++j)
      result[j].store(U[j].get_values() + i);
  }


  /*
   * Convenience wrapper that creates a dealii::Function object out of a
   * (fairly general) callable object:
   */


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
