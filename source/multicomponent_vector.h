//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace ryujin
{
  namespace Vectors
  {
    /**
     * This function takes a scalar MPI partitioner @p scalar_partitioner as
     * argument and returns a shared pointer to a new "vector" multicomponent
     * partitioner that defines storage and MPI synchronization for a vector
     * consisting of @p n_components components. The vector partitioner is
     * intended to efficiently store non-scalar vectors such as the state
     * vectors U. Let (U_i)_k denote the k-th component of a state vector
     * element U_i, we then store
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
     * that stores a vector element of @p n_components components per entry
     * (instead of a scalar value).
     *
     * @ingroup SIMD
     */
    template <typename Number,
              int n_components,
              int simd_length = dealii::VectorizedArray<Number>::size()>
    class MultiComponentVector
        : private dealii::LinearAlgebra::distributed::Vector<Number>
    {
    public:
      /**
       * @name Typedefs and constexpr constants
       */
      //@{

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
      using ScalarHostVector =
          dealii::LinearAlgebra::distributed::Vector<Number>;

      //@}
      /**
       * @name Constructor and reinitialization
       */
      //@{

      /**
       * Default constructor
       */
      MultiComponentVector() = default;

      /**
       * Reinitializes the MultiComponentVector with a vector MPI partitioner
       * that was created first with create_vector_partitioner().
       */
      void reinit_with_vector_partitioner(
          const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
              &vector_partitioner);

      /**
       * Reinitializes the MultiComponentVector with a scalar MPI partitioner.
       * The function calls create_vector_partitioner() internally to
       * create and store a corresponding "vector" MPI partitioner.
       */
      void reinit_with_scalar_partitioner(
          const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
              &scalar_partitioner);

      //@}
      /**
       * @name Extracting and inserting a single component stored in a
       * ScalarHostVector
       */
      //@{

      /**
       * Extracts a single component out of the MultiComponentVector and
       * stores it in @p scalar_vector. The destination vector must have a
       * compatible corresponding (scalar) MPI partitioner, i.e., the "local
       * size", the number of locally owned elements, has to match.
       *
       * The function calls scalar_vector.update_ghost_values() before
       * returning.
       *
       * Optionally, a third argument @p functor can be supplied that is
       * applied to each (scalar) value individually before stored in
       * @p scalar_vector.
       *
       * @note This function is used in the VTUOutput module to unpack a
       * single component out of our custom MultiComponentVector in order to
       * call deal.II specific functions (that can only operate on scalar
       * vectors).
       */
      template <typename Functor = std::identity>
      void extract_component(ScalarHostVector &scalar_vector,
                             unsigned int component,
                             const Functor &functor = std::identity{}) const;

      /**
       * Inserts a single component into a MultiComponentVector. The source
       * vector must have a compatible corresponding (scalar) MPI
       * partitioner, i.e., the "local size", the number of locally owned
       * elements, has to match.
       *
       * The function does not call update_ghost_values() automatically. This
       * has to be done by the user once all components are updated.
       *
       * Optionally, a third argument @p functor can be supplied that is
       * applied to each (scalar) value individually before stored in the
       * corresponding component.
       *
       * @note This function is used in InitialValues to populate all
       * components of the initial state that are returned component wise as
       * single scalar vectors by deal.II interpolation functions.
       */
      template <typename Functor = std::identity>
      void insert_component(const ScalarHostVector &scalar_vector,
                            unsigned int component,
                            const Functor &functor = std::identity{});

      /**
       * Variant of the method above that reads values out of a
       * dealii::Vector in (rank-) local numbering.
       */
      template <typename Functor = std::identity>
      void insert_component(const dealii::Vector<Number> &scalar_vector,
                            unsigned int component,
                            const Functor &functor = std::identity{});

      //@}
      /**
       * @name Access to scalar or tensor-valued entries from various
       * contexts (scalar, SIMD, GPU):
       */
      //@{

      /**
       * Return the entry indexed by @p i.
       *
       * If the template parameter @a Number2 is a VectorizedArray then
       * the function returns a SIMD vectorized dealii::Tensor populated with
       * entries from the @p n_components component vectors stored at
       * indices i, i+1, ..., i+simd_length-1.
       *
       * @note This function is only available if `n_components` is equal to 1.
       */
      template <typename Number2 = Number>
      Number2 read_entry(const unsigned int i) const;

      /**
       * Variant of above function.
       *
       * Returns a SIMD vectorized dealii::Tensor populated with entries from
       * the @p n_components component vectors stored at indices *(js), *(js+1),
       * ..., *(js+simd_length-1), i.e., @p js has to point to an array of
       * size @p simd_length containing all indices.
       *
       * @note This function is only available if `n_components` is equal to 1.
       */
      template <typename Number2 = Number>
      Number2 read_entry(const unsigned int *js) const;

      /**
       * Return the tensor-valued entry indexed by @p i.
       *
       * If the template parameter @a Number2 is a VectorizedArray then
       * the function returns a SIMD vectorized dealii::Tensor populated with
       * entries from the @p n_compontens component vectors stored at
       * indices i, i+1, ..., i+simd_length-1.
       */
      template <typename Number2 = Number,
                typename Tensor = dealii::Tensor<1, n_components, Number2>>
      Tensor read_tensor(const unsigned int i) const;

      /**
       * Variant of above function.
       *
       * Returns a SIMD vectorized dealii::Tensor populated with entries from
       * the @p n_components component vectors stored at indices *(js), *(js+1),
       * ..., *(js+simd_length-1), i.e., @p js has to point to an array of
       * size @p simd_length containing all indices.
       */
      template <typename Number2 = Number,
                typename Tensor = dealii::Tensor<1, n_components, Number2>>
      Tensor read_tensor(const unsigned int *js) const;

      /**
       * Write a (scalar valued) @p entry to the vector at position by @p i.
       *
       * If the template parameter @a Number2 is a VectorizedArray then
       * the function takes a SIMD vectorized @p tensor as argument instead
       * and updates the values of the @p n_components component vectors at
       * indices i, i+1, ..., i+simd_length_1. with the values supplied by @p
       * tensor.
       *
       * @note This function is only available if `n_components` is equal to 1.
       */
      template <typename Number2 = Number>
      void write_entry(const Number2 &entry, const unsigned int i);

      /**
       * Update the values of the @p n_components component vector at index
       * @p i with the values supplied by @p tensor.
       *
       * If the template parameter @a Number2 is a VectorizedArray then
       * the function takes a SIMD vectorized @p tensor as argument instead
       * and updates the values of the @p n_components component vectors at
       * indices i, i+1, ..., i+simd_length_1. with the values supplied by @p
       * tensor.
       *
       * @note @p tensor can be an arbitrary indexable container, such as
       * dealii::Tensor or std::array, that has an `operator[]()` returning a @p
       * Number, and has a type trait `value_type`.
       */
      template <typename Number2 = Number,
                typename Tensor = dealii::Tensor<1, n_components, Number2>>
      void write_tensor(const Tensor &tensor, const unsigned int i);

      /**
       * Add a (scalar valued) @p entry to the vector at position @p i.
       * Update the values of the @p n_components component vector at index @p i
       * by adding the values supplied by @p tensor.
       *
       * If the template parameter @a Number2 is a VectorizedArray then
       * the function takes a SIMD vectorized @p tensor as argument instead
       * and updates the values of the @p n_components component vectors at
       * indices i, i+1, ..., i+simd_length_1. with the values supplied by @p
       * tensor.
       *
       * @note This function is only available if `n_components` is equal to 1.
       */

      template <typename Number2 = Number>
      void add_entry(const Number2 &entry, const unsigned int i);

      /**
       * Update the values of the @p n_components component vector at index @p i
       * by adding the values supplied by @p tensor.
       *
       * If the template parameter @a Number2 is a VectorizedArray then
       * the function takes a SIMD vectorized @p tensor as argument instead
       * and updates the values of the @p n_components component vectors at
       * indices i, i+1, ..., i+simd_length_1. with the values supplied by @p
       * tensor.
       *
       * @note @p tensor can be an arbitrary indexable container, such as
       * dealii::Tensor or std::array, that has an `operator[]()` returning a @p
       * Number, and has a type trait `value_type`.
       */
      template <typename Number2 = Number,
                typename Tensor = dealii::Tensor<1, n_components, Number2>>
      void add_tensor(const Tensor &tensor, const unsigned int i);

      //@}
      /**
       * @name Vector interface
       */
      //@{

      void sadd(const Number s,
                const Number a,
                const MultiComponentVector<Number, n_components> &V)
      {
        ScalarHostVector::sadd(s, a, V);
      }

      using ScalarHostVector::update_ghost_values;

      using ScalarHostVector::zero_out_ghost_values;

      using ScalarHostVector::compress;

      //@}
    };


#ifndef DOXYGEN
    /*
     * -------------------------------------------------------------------------
     * Inline function definitions
     * -------------------------------------------------------------------------
     */


    template <typename Number, int n_components, int simd_length>
    void MultiComponentVector<Number, n_components, simd_length>::
        reinit_with_vector_partitioner(
            const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
                &vector_partitioner)
    {
      /* Special case of a zero component vector */
      if (n_components == 0)
        return;

      ScalarHostVector::reinit(vector_partitioner);
    }

    template <typename Number, int n_components, int simd_length>
    void MultiComponentVector<Number, n_components, simd_length>::
        reinit_with_scalar_partitioner(
            const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
                &scalar_partitioner)
    {
      /* Special case of a zero component vector: */
      if (n_components == 0)
        return;

      /* Special case of a scalar vector: */
      if (n_components == 1)
        ScalarHostVector::reinit(scalar_partitioner);

      auto vector_partitioner =
          create_vector_partitioner(scalar_partitioner, n_components);

      ScalarHostVector::reinit(vector_partitioner);
    }


    template <typename Number, int n_components, int simd_length>
    template <typename Functor>
    void
    MultiComponentVector<Number, n_components, simd_length>::extract_component(
        ScalarHostVector &scalar_vector,
        unsigned int component,
        const Functor &functor) const
    {
      Assert(n_components > 0,
             dealii::ExcMessage(
                 "Cannot extract from a vector with zero components."));
      AssertIndexRange(component, n_components);

      Assert(n_components *
                     scalar_vector.get_partitioner()->locally_owned_size() ==
                 this->get_partitioner()->locally_owned_size(),
             dealii::ExcMessage("Called with a scalar_vector argument that has "
                                "incompatible local range."));
      const auto local_size =
          scalar_vector.get_partitioner()->locally_owned_size();
      for (unsigned int i = 0; i < local_size; ++i)
        scalar_vector.local_element(i) =
            functor(this->local_element(i * n_components + component));
      scalar_vector.update_ghost_values();
    }


    template <typename Number, int n_components, int simd_length>
    template <typename Functor>
    void
    MultiComponentVector<Number, n_components, simd_length>::insert_component(
        const ScalarHostVector &scalar_vector,
        unsigned int component,
        const Functor &functor)
    {
      Assert(n_components > 0,
             dealii::ExcMessage(
                 "Cannot insert into a vector with zero components."));
      AssertIndexRange(component, n_components);

      Assert(n_components *
                     scalar_vector.get_partitioner()->locally_owned_size() ==
                 this->get_partitioner()->locally_owned_size(),
             dealii::ExcMessage("Called with a scalar_vector argument that has "
                                "incompatible local range."));
      const auto local_size =
          scalar_vector.get_partitioner()->locally_owned_size();
      for (unsigned int i = 0; i < local_size; ++i)
        this->local_element(i * n_components + component) =
            functor(scalar_vector.local_element(i));
    }


    template <typename Number, int n_components, int simd_length>
    template <typename Functor>
    void
    MultiComponentVector<Number, n_components, simd_length>::insert_component(
        const dealii::Vector<Number> &scalar_vector,
        unsigned int component,
        const Functor &functor)
    {
      Assert(n_components > 0,
             dealii::ExcMessage(
                 "Cannot insert into a vector with zero components."));
      AssertIndexRange(component, n_components);

      Assert(n_components * scalar_vector.size() >=
                 this->get_partitioner()->locally_owned_size(),
             dealii::ExcMessage("Called with a scalar_vector argument that has "
                                "incompatible local range."));
      const auto local_size = scalar_vector.size();
      for (unsigned int i = 0; i < local_size; ++i)
        this->local_element(i * n_components + component) =
            functor(scalar_vector[i]);
    }


    template <typename Number, int n_components, int simd_length>
    template <typename Number2>
    DEAL_II_ALWAYS_INLINE inline Number2
    MultiComponentVector<Number, n_components, simd_length>::read_entry(
        const unsigned int i) const
    {
      static_assert(
          n_components == 1,
          "Attempted to read a scalar value from a tensor-valued vector entry");

      AssertIndexRange(i,
                       this->get_partitioner()->locally_owned_size() +
                           this->get_partitioner()->n_ghost_indices());

      const auto result = read_tensor<Number2>(i);
      return result[0];
    }


    template <typename Number, int n_components, int simd_length>
    template <typename Number2, typename Tensor>
    DEAL_II_ALWAYS_INLINE inline Tensor
    MultiComponentVector<Number, n_components, simd_length>::read_tensor(
        const unsigned int i) const
    {
      static_assert(std::is_same_v<Number2, typename Tensor::value_type>,
                    "type mismatch");

      AssertIndexRange(i,
                       this->get_partitioner()->locally_owned_size() +
                           this->get_partitioner()->n_ghost_indices());

      Tensor tensor;

      /* Special case of a zero component vector */
      if constexpr (n_components == 0)
        return tensor;

      if constexpr (std::is_same_v<VectorizedArray, Number2>) {
        /* Vectorized fast access. index must be divisible by simd_length */
        std::array<unsigned int, VectorizedArray::size()> indices;
        for (unsigned int k = 0; k < VectorizedArray::size(); ++k)
          indices[k] = k * n_components;

        dealii::vectorized_load_and_transpose(n_components,
                                              this->begin() + i * n_components,
                                              indices.data(),
                                              &tensor[0]);
      } else {
        /* Non-vectorized sequential access. */

        for (unsigned int d = 0; d < n_components; ++d)
          tensor[d] = this->local_element(i * n_components + d);
      }

      return tensor;
    }


    template <typename Number, int n_components, int simd_length>
    template <typename Number2>
    DEAL_II_ALWAYS_INLINE inline Number2
    MultiComponentVector<Number, n_components, simd_length>::read_entry(
        const unsigned int *js) const
    {
      static_assert(
          n_components == 1,
          "Attempted to read a scalar value from a tensor-valued vector entry");

      const auto result = read_tensor<Number2>(js);
      return result[0];
    }


    template <typename Number, int n_components, int simd_length>
    template <typename Number2, typename Tensor>
    DEAL_II_ALWAYS_INLINE inline Tensor
    MultiComponentVector<Number, n_components, simd_length>::read_tensor(
        const unsigned int *js) const
    {
      static_assert(std::is_same_v<Number2, typename Tensor::value_type>,
                    "type mismatch");
      Tensor tensor;

      /* Special case of a zero component vector */
      if constexpr (n_components == 0)
        return tensor;

      if constexpr (std::is_same_v<VectorizedArray, Number2>) {
        /* Vectorized fast access. index must be divisible by simd_length */

        std::array<unsigned int, VectorizedArray::size()> indices;
        for (unsigned int k = 0; k < VectorizedArray::size(); ++k) {
          AssertIndexRange(js[k],
                           this->get_partitioner()->locally_owned_size() +
                               this->get_partitioner()->n_ghost_indices());
          indices[k] = js[k] * n_components;
        }

        dealii::vectorized_load_and_transpose(
            n_components, this->begin(), indices.data(), &tensor[0]);

      } else {
        /* Non-vectorized sequential access. */

        AssertIndexRange(*js,
                         this->get_partitioner()->locally_owned_size() +
                             this->get_partitioner()->n_ghost_indices());

        for (unsigned int d = 0; d < n_components; ++d)
          tensor[d] = this->local_element(js[0] * n_components + d);
      }

      return tensor;
    }


    template <typename Number, int n_components, int simd_length>
    template <typename Number2>
    DEAL_II_ALWAYS_INLINE inline void
    MultiComponentVector<Number, n_components, simd_length>::write_entry(
        const Number2 &entry, const unsigned int i)
    {
      static_assert(n_components == 1,
                    "Attempted to write a scalar value into a tensor-valued "
                    "vector entry");

      AssertIndexRange(i,
                       this->get_partitioner()->locally_owned_size() +
                           this->get_partitioner()->n_ghost_indices());

      dealii::Tensor<1, n_components, Number2> tensor;
      tensor[0] = entry;

      write_tensor<Number2>(tensor, i);
    }


    template <typename Number, int n_components, int simd_length>
    template <typename Number2, typename Tensor>
    DEAL_II_ALWAYS_INLINE inline void
    MultiComponentVector<Number, n_components, simd_length>::write_tensor(
        const Tensor &tensor, const unsigned int i)
    {
      static_assert(std::is_same_v<Number2, typename Tensor::value_type>,
                    "type mismatch");

      AssertIndexRange(i,
                       this->get_partitioner()->locally_owned_size() +
                           this->get_partitioner()->n_ghost_indices());

      /* Special case of a zero component vector */
      if constexpr (n_components == 0)
        return;

      if constexpr (std::is_same_v<VectorizedArray, Number2>) {
        /* Vectorized fast access. index must be divisible by simd_length */

        std::array<unsigned int, VectorizedArray::size()> indices;
        for (unsigned int k = 0; k < VectorizedArray::size(); ++k)
          indices[k] = k * n_components;

        dealii::vectorized_transpose_and_store(/*add into*/ false,
                                               n_components,
                                               &tensor[0],
                                               indices.data(),
                                               this->begin() +
                                                   i * n_components);

      } else {
        /* Non-vectorized sequential access. */

        for (unsigned int d = 0; d < n_components; ++d)
          this->local_element(i * n_components + d) = tensor[d];
      }
    }


    template <typename Number, int n_components, int simd_length>
    template <typename Number2>
    DEAL_II_ALWAYS_INLINE inline void
    MultiComponentVector<Number, n_components, simd_length>::add_entry(
        const Number2 &entry, const unsigned int i)
    {
      static_assert(n_components == 1,
                    "Attempted to write a scalar value into a tensor-valued "
                    "matrix entry");

      AssertIndexRange(i,
                       this->get_partitioner()->locally_owned_size() +
                           this->get_partitioner()->n_ghost_indices());

      dealii::Tensor<1, n_components, Number2> tensor;
      tensor[0] = entry;

      add_tensor<Number2>(tensor, i);
    }


    template <typename Number, int n_components, int simd_length>
    template <typename Number2, typename Tensor>
    DEAL_II_ALWAYS_INLINE inline void
    MultiComponentVector<Number, n_components, simd_length>::add_tensor(
        const Tensor &tensor, const unsigned int i)
    {
      static_assert(std::is_same_v<Number2, typename Tensor::value_type>,
                    "type mismatch");

      AssertIndexRange(i,
                       this->get_partitioner()->locally_owned_size() +
                           this->get_partitioner()->n_ghost_indices());

      /* Special case of a zero component vector */
      if constexpr (n_components == 0)
        return;

      if constexpr (std::is_same_v<VectorizedArray, Number2>) {
        /* Vectorized fast access. index must be divisible by simd_length */

        std::array<unsigned int, VectorizedArray::size()> indices;
        for (unsigned int k = 0; k < VectorizedArray::size(); ++k)
          indices[k] = k * n_components;

        dealii::vectorized_transpose_and_store(/*add into*/ true,
                                               n_components,
                                               &tensor[0],
                                               indices.data(),
                                               this->begin() +
                                                   i * n_components);

      } else {
        /* Non-vectorized sequential access. */

        for (unsigned int d = 0; d < n_components; ++d)
          this->local_element(i * n_components + d) += tensor[d];
      }
    }

#endif
  } // namespace Vectors
} // namespace ryujin
