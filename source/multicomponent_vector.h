//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "simd.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/la_parallel_vector.h>

#if DEAL_II_VERSION_GTE(9, 3, 0)
#define LOCAL_SIZE locally_owned_size
#else
#define LOCAL_SIZE local_size
#endif

namespace ryujin
{
  /**
   * This function takes a scalar MPI partitioner @p scalar_partitioner as
   * argument and returns a shared pointer to a new "vector" multicomponent
   * partitioner that defines storage and MPI synchronization for a vector
   * consisting of @p n_comp components. The vector partitioner is intended
   * to efficiently store non-scalar vectors such as the state vectors U.
   * Let (U_i)_k denote the k-th component of a state vector element U_i,
   * we then store
   * \f{align}
   *  (U_0)_0, (U_0)_1, (U_0)_2, (U_0)_3, (U_0)_4,
   *  (U_1)_0, (U_1)_1, (U_1)_2, (U_1)_3, (U_1)_4,
   *  \ldots
   * \f}
   *
   * @note This function is used to efficiently set up a single vector
   * partitioner in OfflineData used in all MultiComponentVector instances.
   *
   * @ingroup SIMD
   */
  std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
  create_vector_partitioner(
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          &scalar_partitioner,
      const unsigned int n_components);


  /**
   * A wrapper around dealii::LinearAlgebra::distributed::Vector<Number>
   * that stores a vector element of @p n_comp components per entry
   * (instead of a scalar value).
   *
   * @note reinit() has to be called with an appropriate "vector" MPI
   * partitioner created by create_vector_partitioner().
   *
   * @ingroup SIMD
   */
  template <typename Number,
            int n_comp,
            int simd_length = dealii::VectorizedArray<Number>::size()>
  class MultiComponentVector
      : public dealii::LinearAlgebra::distributed::Vector<Number>
  {
  public:
    /**
     * Shorthand typedef for the underlying dealii::VectorizedArray type
     * used to insert and extract SIMD packed values from the
     * MultiComponentVector.
     */
    using VectorizedArray = dealii::VectorizedArray<Number, simd_length>;

    /**
     * Shorthand typedef for the underlying scalar
     * dealii::LinearAlgebra::distributed::Vector<Number> used to insert
     * and extract a single component of the MultiComponentVector.
     */
    using scalar_type = dealii::LinearAlgebra::distributed::Vector<Number>;

    /**
     * We want to use the assignment operator of the virtual base class, so
     * specify that here.
     */
    using scalar_type::operator=;

    /**
     * Reinitializes the MultiComponentVector with a scalar MPI
     * partitioner. The function calls create_vector_partitioner()
     * internally to create and store a corresponding "vector" MPI
     * partitioner.
     */
    void reinit_with_scalar_partitioner(
        const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
            &scalar_partitioner);

    /**
     * Extracts a single component out of the MultiComponentVector and
     * stores it in @p scalar_vector. The destination vector must have a
     * compatible corresponding (scalar) MPI partitioner, i.e., the "local
     * size", the number of locally owned elements, has to match.
     *
     * The function calls scalar_vector.update_ghost_values() before
     * returning.
     *
     * @note This function is used in the VTUOutput module to unpack a
     * single component out of our custom MultiComponentVector in order to
     * call deal.II specific functions (that can only operate on scalar
     * vectors).
     */
    void extract_component(scalar_type &scalar_vector,
                           unsigned int component) const;

    /**
     * Inserts a single component into a MultiComponentVector. The source
     * vector must have a compatible corresponding (scalar) MPI
     * partitioner, i.e., the "local size", the number of locally owned
     * elements, has to match.
     *
     * The function does not call update_ghost_values() automatically. This
     * has to be done by the user once all components are updated.
     *
     * @note This function is used in InitialValues to populate all
     * components of the initial state that are returned component wise as
     * single scalar vectors by deal.II interpolation functions.
     */
    void insert_component(const scalar_type &scalar_vector,
                          unsigned int component);

    /**
     * Return a dealii::Tensor populated with the @p n_comp component
     * vector stored at index @p i.
     *
     * If the template parameter @tref Number2 is a VectorizedArray then
     * the function returns a SIMD vectorized dealii::Tensor populated with
     * entries from the @p n_comp component vectors stored at indices i,
     * i+1, ..., i+simd_length-1.
     */
    template <typename Number2 = Number,
              typename Tensor = dealii::Tensor<1, n_comp, Number2>>
    Tensor get_tensor(const unsigned int i) const;

    /**
     * Variant of above function.
     *
     * Returns a SIMD vectorized dealii::Tensor populated with entries from
     * the @p n_comp component vectors stored at indices *(js), *(js+1),
     * ..., *(js+simd_length-1), i.e., @p js has to point to an array of
     * size @p simd_length containing all indices.
     */
    template <typename Number2 = Number,
              typename Tensor = dealii::Tensor<1, n_comp, Number2>>
    Tensor get_tensor(const unsigned int *js) const;

    /**
     * Update the values of the @p n_comp component vector at index @p i
     * with the values supplied by @p tensor.
     *
     * If the template parameter @tref Number2 is a VectorizedArray then
     * the function takes a SIMD vectorized @p tensor as argument instead
     * and updates the values of the @p n_comp component vectors at indices
     * i, i+1, ..., i+simd_length_1. with the values supplied by @p tensor.
     *
     * @note @p tensor can be an arbitrary indexable container, such as
     * dealii::Tensor or std::array, that has an `operator[]()` returning a @p
     * Number, and has a type trait `value_type`.
     */
    template <typename Number2 = Number,
              typename Tensor = dealii::Tensor<1, n_comp, Number2>>
    void write_tensor(const Tensor &tensor, const unsigned int i);
  };


#ifndef DOXYGEN
  /* Template definitions: */

  template <typename Number, int n_comp, int simd_length>
  void MultiComponentVector<Number, n_comp, simd_length>::
      reinit_with_scalar_partitioner(
          const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
              &scalar_partitioner)
  {
    /* Special case of a zero component vector */
    if (n_comp == 0)
      return;

    auto vector_partitioner =
        create_vector_partitioner(scalar_partitioner, n_comp);

    dealii::LinearAlgebra::distributed::Vector<Number>::reinit(
        vector_partitioner);
  }


  template <typename Number, int n_comp, int simd_length>
  void MultiComponentVector<Number, n_comp, simd_length>::extract_component(
      scalar_type &scalar_vector, unsigned int component) const
  {
    Assert(n_comp > 0,
           dealii::ExcMessage(
               "Cannot extract from a vector with zero components."));

    Assert(n_comp * scalar_vector.get_partitioner()->LOCAL_SIZE() ==
               this->get_partitioner()->LOCAL_SIZE(),
           dealii::ExcMessage("Called with a scalar_vector argument that has "
                              "incompatible local range."));
    const auto local_size = scalar_vector.get_partitioner()->LOCAL_SIZE();
    for (unsigned int i = 0; i < local_size; ++i)
      scalar_vector.local_element(i) =
          this->local_element(i * n_comp + component);
    scalar_vector.update_ghost_values();
  }


  template <typename Number, int n_comp, int simd_length>
  void MultiComponentVector<Number, n_comp, simd_length>::insert_component(
      const scalar_type &scalar_vector, unsigned int component)
  {
    Assert(n_comp > 0,
           dealii::ExcMessage(
               "Cannot insert into a vector with zero components."));

    Assert(n_comp * scalar_vector.get_partitioner()->LOCAL_SIZE() ==
               this->get_partitioner()->LOCAL_SIZE(),
           dealii::ExcMessage("Called with a scalar_vector argument that has "
                              "incompatible local range."));
    const auto local_size = scalar_vector.get_partitioner()->LOCAL_SIZE();
    for (unsigned int i = 0; i < local_size; ++i)
      this->local_element(i * n_comp + component) =
          scalar_vector.local_element(i);
  }

  /* Inline function  definitions: */

  template <typename Number, int n_comp, int simd_length>
  template <typename Number2, typename Tensor>
  DEAL_II_ALWAYS_INLINE inline Tensor
  MultiComponentVector<Number, n_comp, simd_length>::get_tensor(
      const unsigned int i) const
  {
    static_assert(std::is_same<Number2, typename Tensor::value_type>::value,
                  "dummy type mismatch");
    Tensor tensor;

    /* Special case of a zero component vector */
    if constexpr (n_comp == 0)
      return tensor;

    if constexpr (std::is_same<Number, Number2>::value) {
      /* Non-vectorized sequential access. */

      for (unsigned int d = 0; d < n_comp; ++d)
        tensor[d] = this->local_element(i * n_comp + d);

    } else if constexpr (std::is_same<VectorizedArray, Number2>::value) {

      /* Vectorized fast access. index must be divisible by simd_length */
      unsigned int indices[VectorizedArray::size()];
      for (unsigned int k = 0; k < VectorizedArray::size(); ++k)
        indices[k] = k * n_comp;

      dealii::vectorized_load_and_transpose(
          n_comp, this->begin() + i * n_comp, indices, &tensor[0]);

    } else {
      /* not implemented */
      __builtin_trap();
    }

    return tensor;
  }


  template <typename Number, int n_comp, int simd_length>
  template <typename Number2, typename Tensor>
  DEAL_II_ALWAYS_INLINE inline Tensor
  MultiComponentVector<Number, n_comp, simd_length>::get_tensor(
      const unsigned int *js) const
  {
    static_assert(std::is_same<Number2, typename Tensor::value_type>::value,
                  "dummy type mismatch");
    Tensor tensor;

    /* Special case of a zero component vector */
    if constexpr (n_comp == 0)
      return tensor;

    if constexpr (std::is_same<Number, Number2>::value) {
      /* Non-vectorized sequential access. */

      for (unsigned int d = 0; d < n_comp; ++d)
        tensor[d] = this->local_element(js[0] * n_comp + d);

    } else if constexpr (std::is_same<VectorizedArray, Number2>::value) {
      /* Vectorized fast access. index must be divisible by simd_length */

      unsigned int indices[VectorizedArray::size()];
      for (unsigned int k = 0; k < VectorizedArray::size(); ++k)
        indices[k] = js[k] * n_comp;

      dealii::vectorized_load_and_transpose(
          n_comp, this->begin(), indices, &tensor[0]);

    } else {
      /* not implemented */
      __builtin_trap();
    }

    return tensor;
  }


  template <typename Number, int n_comp, int simd_length>
  template <typename Number2, typename Tensor>
  DEAL_II_ALWAYS_INLINE inline void
  MultiComponentVector<Number, n_comp, simd_length>::write_tensor(
      const Tensor &tensor, const unsigned int i)
  {
    static_assert(std::is_same<Number2, typename Tensor::value_type>::value,
                  "dummy type mismatch");

    /* Special case of a zero component vector */
    if constexpr (n_comp == 0)
      return;

    if constexpr (std::is_same<Number, Number2>::value) {
      /* Non-vectorized sequential access. */

      for (unsigned int d = 0; d < n_comp; ++d)
        this->local_element(i * n_comp + d) = tensor[d];

    } else if constexpr (std::is_same<VectorizedArray, Number2>::value) {
      /* Vectorized fast access. index must be divisible by simd_length */

      unsigned int indices[VectorizedArray::size()];
      for (unsigned int k = 0; k < VectorizedArray::size(); ++k)
        indices[k] = k * n_comp;

      dealii::vectorized_transpose_and_store(
          false, n_comp, &tensor[0], indices, this->begin() + i * n_comp);

    } else {
      /* not implemented */
      __builtin_trap();
    }
  }
#endif

} // namespace ryujin

#undef LOCAL_SIZE
