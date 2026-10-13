//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "gpu.h"
#include "loop.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace ryujin
{
  /**
   * This function takes a scalar MPI partitioner @p scalar_partitioner as
   * argument and returns a shared pointer to a new "vector" multicomponent
   * partitioner that defines storage and MPI synchronization for a vector
   * consisting of @p n_comp components. The vector partitioner is
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
   * @ingroup LinearAlgebra
   */
  std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
  create_vector_partitioner(
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          &scalar_partitioner,
      const unsigned int n_comp);


  template <typename Number,
            int n_comp,
            typename MemorySpace = dealii::MemorySpace::Host,
            bool writable = true>
  class MultiComponentVectorView;


  /**
   * A wrapper around dealii::LinearAlgebra::distributed::Vector<Number>
   * that stores a vector element of n_comp() components per entry
   * (instead of a scalar value).
   *
   * @ingroup LinearAlgebra
   */
  template <typename Number>
  class MultiComponentVector
      : public MirroredStorage<MultiComponentVector<Number>>
  {
  public:
    /**
     * @name Constructor and initialization
     */
    //@{

    MultiComponentVector() = default;

    MultiComponentVector(const MultiComponentVector &other);

    MultiComponentVector(MultiComponentVector &&other) noexcept;

    /**
     * Reinitializes the MultiComponentVector with a vector MPI partitioner
     * that was created first with create_vector_partitioner() for @p n_comp
     * components.
     *
     * After reinit the vector is resident on the host memory space only;
     * device storage is allocated lazily on the first
     * copy_to_memory_space() / move_to_memory_space(). (If the host and
     * default memory spaces coincide the vector is resident on both.)
     *
     * @note Because the vector ends up resident on the host memory space
     * only, TransferPolicy::implicit_transfers_default_resident is not an
     * admissible @p transfer_policy for this function.
     */
    void reinit_with_vector_partitioner(
        const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
            &vector_partitioner,
        const unsigned int n_comp,
        const TransferPolicy transfer_policy =
            TransferPolicy::explicit_transfers);

    /**
     * Reinitializes the MultiComponentVector with a scalar MPI partitioner
     * for @p n_comp components. The function calls
     * create_vector_partitioner() internally to create and store a
     * corresponding "vector" MPI partitioner.
     *
     * After reinit the vector is resident on the host memory space only;
     * device storage is allocated lazily on the first
     * copy_to_memory_space() / move_to_memory_space(). (If the host and
     * default memory spaces coincide the vector is resident on both.)
     *
     * @note Because the vector ends up resident on the host memory space
     * only, TransferPolicy::implicit_transfers_default_resident is not an
     * admissible @p transfer_policy for this function.
     */
    void reinit_with_scalar_partitioner(
        const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
            &scalar_partitioner,
        const unsigned int n_comp,
        const TransferPolicy transfer_policy =
            TransferPolicy::explicit_transfers);

    /**
     * Return the number of components per entry.
     */
    ACCESSOR_READ_ONLY(n_comp)

    MultiComponentVector &operator=(const MultiComponentVector &other);

    MultiComponentVector &operator=(MultiComponentVector &&other) noexcept;

    //@}
    /**
     * @name Memory space access and synchronization
     */
    //@{

    /**
     * Return a writable view on the vector for the selected memory
     * space. Depending on the selected TransferPolicy the method either
     * asserts that the memory space is resident
     * (TransferPolicy::explicit_transfers), or performs an implicit
     * move_to_memory_space() invalidating the other memory space
     * (TransferPolicy::implicit_transfers).
     *
     * The template parameter @p n_comp has to match n_comp().
     */
    template <int n_comp, typename MemorySpace = dealii::MemorySpace::Host>
    MultiComponentVectorView<Number, n_comp, MemorySpace, true> view();

    /**
     * Return a read-only view on the vector for the selected memory
     * space. Depending on the selected TransferPolicy the method either
     * asserts that the memory space is resident
     * (TransferPolicy::explicit_transfers), or performs an implicit
     * copy_to_memory_space() (TransferPolicy::implicit_transfers).
     *
     * The template parameter @p n_comp has to match n_comp().
     */
    template <int n_comp, typename MemorySpace = dealii::MemorySpace::Host>
    MultiComponentVectorView<Number, n_comp, MemorySpace, false> view() const;

    /**
     * Return a reference to the underlying deal.II vector residing in
     * the selected memory space.
     *
     * @note Only for n_comp() == 1 is the deal.II vector compatible with
     * the scalar DoFHandler.
     *
     * @note The same residency and transfer semantics as for the writable
     * view() apply.
     */
    template <typename MemorySpace = dealii::MemorySpace::Host>
    auto &deal_ii_vector();

    /**
     * Return a const reference to the underlying deal.II vector residing
     * in the selected memory space, see above.
     *
     * @note The same residency and transfer semantics as for the
     * read-only view() apply.
     */
    template <typename MemorySpace = dealii::MemorySpace::Host>
    const auto &deal_ii_vector() const;

    /*
     * The is_resident(), copy_to_memory_space(), move_to_memory_space(),
     * transfer_policy(), and set_transfer_policy() methods are inherited
     * from the MirroredStorage base class.
     */

  private:
    //@}
    /**
     * @name Internal fields, methods, and friends
     */
    //@{

    unsigned int n_comp_ = 0;

    /*
     * Note: The storage is marked mutable so that the (logically const)
     * copy_to_memory_space() operation can populate a mirror from within
     * a const view() under the implicit_transfers policy.
     *
     * We avoid setting up the default_vector_ if default and host happen
     * to be the same memory space (see have_separate_memory_spaces).
     */

    mutable dealii::LinearAlgebra::distributed::
        Vector<Number, dealii::MemorySpace::Host>
            host_vector_;

    mutable dealii::LinearAlgebra::distributed::
        Vector<Number, dealii::MemorySpace::Default>
            default_vector_;

    /*
     * Return the deal.II vector holding the data of the selected memory
     * space:
     */
    template <typename MemorySpace>
    auto &storage() const;

    /*
     * Storage primitives used by the MirroredStorage base class:
     */

    template <typename MemorySpace>
    void allocate_storage() const;

    template <typename To, typename From>
    void deep_copy_storage() const;

    template <typename MemorySpace>
    void deallocate_storage();


    friend class MirroredStorage<MultiComponentVector<Number>>;

    template <typename, int, typename, bool>
    friend class MultiComponentVectorView;

    //@}
  };


  /**
   * A "view" of a MultiComponentVector that lives in the host or device
   * memory space. It provides a number of methods for reading and
   * writing scalar and tensor-valued entries.
   *
   * @note This class is designed to be captured by value in computation
   * loops with access to either the host or device memory space. As such
   * we do not store a reference to the underlying MultiComponentVector
   * but rather raw pointers into the corresponding memory. The view is
   * only valid as long as the underlying MultiComponentVector object is
   * not modified.
   *
   * @ingroup LinearAlgebra
   */
  template <typename Number, int n_comp, typename MemorySpace, bool writable>
  class MultiComponentVectorView
  {
  public:
    /**
     * @name Constructor and initialization
     */
    //@{

    MultiComponentVectorView() = default;

    MultiComponentVectorView(
        MultiComponentVector<Number> &multi_component_vector)
      requires(writable)
    {
      reinit(multi_component_vector);
    }

    MultiComponentVectorView(
        const MultiComponentVector<Number> &multi_component_vector)
      requires(!writable)
    {
      reinit(multi_component_vector);
    }

    /**
     * Converting constructor creating a read only view from a writable
     * view (mirroring the conversion from `T *` to `const T *`).
     *
     * @note We need to make this a templated constructor, otherwise the
     * writable-converting constructor here would suppress the default
     * copy constructor.
     */
    template <bool other_writable>
    DEAL_II_HOST_DEVICE MultiComponentVectorView(
        const MultiComponentVectorView<Number,
                                       n_comp,
                                       MemorySpace,
                                       other_writable> &other)
      requires(!writable && other_writable)
        : multi_component_vector_(other.multi_component_vector_)
        , data_(other.data_)
        , n_locally_owned_(other.n_locally_owned_)
        , n_locally_relevant_(other.n_locally_relevant_)
    {
    }

    template <typename MultiComponentVector>
    void reinit(MultiComponentVector &multi_component_vector)
      requires(writable != std::is_const_v<MultiComponentVector>)
    {
      Assert(multi_component_vector.n_comp() ==
                 static_cast<unsigned int>(n_comp),
             dealii::ExcMessage("The number of components of the view does "
                                "not match the number of components of the "
                                "vector."));

      multi_component_vector_ = &multi_component_vector;

      auto &vector = multi_component_vector_->template storage<MemorySpace>();
      const auto &partitioner = vector.get_partitioner();

      data_ = vector.begin();
      n_locally_owned_ = partitioner->locally_owned_size();
      n_locally_relevant_ = n_locally_owned_ + partitioner->n_ghost_indices();
    }

    //@}
    /**
     * @name Typedefs and constexpr constants
     */
    //@{

    /**
     * Shorthand typedef for the underlying scalar deal.II vector used to
     * insert and extract a single component of the MultiComponentVector.
     */
    using ScalarVector =
        dealii::LinearAlgebra::distributed::Vector<Number, MemorySpace>;

    //@}
    /**
     * @name Extracting and inserting components
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
     * @note The operation is performed on the memory space of the view.
     * Correspondingly, @p functor has to be callable on the selected
     * memory space.
     *
     * @note This function is used in the VTUOutput module to unpack a
     * single component out of our custom MultiComponentVector in order to
     * call deal.II specific functions (that can only operate on scalar
     * vectors).
     */
    template <typename Functor = std::identity>
    void extract_component(ScalarVector &scalar_vector,
                           unsigned int component,
                           const Functor &functor = std::identity{}) const
    {
      Assert(n_comp > 0,
             dealii::ExcMessage(
                 "Cannot extract from a vector with zero components."));
      AssertIndexRange(component, n_comp);

      const auto local_size = static_cast<unsigned int>(
          scalar_vector.get_partitioner()->locally_owned_size());

      Assert(n_comp * local_size == n_locally_owned_,
             dealii::ExcMessage("Called with a scalar_vector argument that has "
                                "incompatible local range."));

      const auto *data = data_;
      auto *destination = scalar_vector.begin();

      const auto body = [=](auto /*sentinel*/, unsigned int i) {
        destination[i] = functor(data[i * n_comp + component]);
      };

      loop<MemorySpace, Number>(
          "extract_component", body, 0, /*no vectorization*/ 0, local_size);

      scalar_vector.update_ghost_values();
    }

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
    void insert_component(const ScalarVector &scalar_vector,
                          unsigned int component,
                          const Functor &functor = std::identity{}) const
      requires writable
    {
      using HostSpace = dealii::MemorySpace::Host;
      AssertThrow((std::is_same_v<MemorySpace, HostSpace>),
                  dealii::ExcNotImplemented());

      Assert(n_comp > 0,
             dealii::ExcMessage(
                 "Cannot insert into a vector with zero components."));
      AssertIndexRange(component, n_comp);

      const auto local_size =
          scalar_vector.get_partitioner()->locally_owned_size();

      Assert(n_comp * local_size == n_locally_owned_,
             dealii::ExcMessage("Called with a scalar_vector argument that has "
                                "incompatible local range."));

      for (unsigned int i = 0; i < local_size; ++i)
        data_[i * n_comp + component] = functor(scalar_vector.local_element(i));
    }

    /**
     * Variant of the method above that reads values out of a
     * dealii::Vector in (rank-) local numbering.
     */
    template <typename Functor = std::identity>
    void insert_component(const dealii::Vector<Number> &scalar_vector,
                          unsigned int component,
                          const Functor &functor = std::identity{}) const
      requires writable
    {
      using HostSpace = dealii::MemorySpace::Host;
      AssertThrow((std::is_same_v<MemorySpace, HostSpace>),
                  dealii::ExcInternalError());

      Assert(n_comp > 0,
             dealii::ExcMessage(
                 "Cannot insert into a vector with zero components."));
      AssertIndexRange(component, n_comp);

      const auto local_size = scalar_vector.size();

      Assert(n_comp * local_size >= n_locally_owned_,
             dealii::ExcMessage("Called with a scalar_vector argument that has "
                                "incompatible local range."));

      for (unsigned int i = 0; i < local_size; ++i)
        data_[i * n_comp + component] = functor(scalar_vector[i]);
    }

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
     * entries from the @p n_comp component vectors stored at
     * indices i, i+1, ..., i+n-1, where n is the SIMD width.
     *
     * @note This function is only available if `n_comp` is equal to 1.
     */
    template <typename Number2 = Number>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number2
    read_entry(const unsigned int i) const
    {
      static_assert(
          n_comp == 1,
          "Attempted to read a scalar value from a tensor-valued vector entry");

      AssertIndexRange(i, n_locally_relevant_);

      const auto result = read_tensor<Number2>(i);
      return result[0];
    }

    /**
     * Variant of above function.
     *
     * Returns a SIMD vectorized dealii::Tensor populated with entries from
     * the @p n_comp component vectors stored at indices *(js), *(js+1),
     * ..., *(js+n-1), i.e., @p js has to point to an array of size n (the
     * SIMD width) containing all indices.
     *
     * @note This function is only available if `n_comp` is equal to 1.
     */
    template <typename Number2 = Number>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number2
    read_entry(const unsigned int *js) const
    {
      static_assert(
          n_comp == 1,
          "Attempted to read a scalar value from a tensor-valued vector entry");

      const auto result = read_tensor<Number2>(js);
      return result[0];
    }

    /**
     * Return the tensor-valued entry indexed by @p i.
     *
     * If the template parameter @a Number2 is a VectorizedArray then
     * the function returns a SIMD vectorized dealii::Tensor populated with
     * entries from the @p n_compontens component vectors stored at
     * indices i, i+1, ..., i+n-1, where n is the SIMD width.
     */
    template <typename Number2 = Number,
              typename Tensor = dealii::Tensor<1, n_comp, Number2>>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Tensor
    read_tensor(const unsigned int i) const
    {
      static_assert(std::is_same_v<Number2, typename Tensor::value_type>,
                    "type mismatch");

      AssertIndexRange(i, n_locally_relevant_);

      Tensor tensor;

      /* Special case of a zero component vector */
      if constexpr (n_comp == 0)
        return tensor;

      using VA = dealii::VectorizedArray<Number>;
      if constexpr (std::is_same_v<VA, Number2>) {
        /* Vectorized fast access. index must be divisible by the SIMD width */
        std::array<unsigned int, VA::size()> indices;
        for (unsigned int k = 0; k < VA::size(); ++k)
          indices[k] = k * n_comp;

        dealii::vectorized_load_and_transpose(
            n_comp, data_ + i * n_comp, indices.data(), &tensor[0]);

      } else {
        /* Non-vectorized sequential access. */
        for (unsigned int d = 0; d < n_comp; ++d)
          tensor[d] = data_[i * n_comp + d];
      }

      return tensor;
    }

    /**
     * Variant of above function.
     *
     * Returns a SIMD vectorized dealii::Tensor populated with entries from
     * the @p n_comp component vectors stored at indices *(js), *(js+1),
     * ..., *(js+n-1), i.e., @p js has to point to an array of size n (the
     * SIMD width) containing all indices.
     */
    template <typename Number2 = Number,
              typename Tensor = dealii::Tensor<1, n_comp, Number2>>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Tensor
    read_tensor(const unsigned int *js) const
    {
      static_assert(std::is_same_v<Number2, typename Tensor::value_type>,
                    "type mismatch");

      Tensor tensor;

      /* Special case of a zero component vector */
      if constexpr (n_comp == 0)
        return tensor;

      using VA = dealii::VectorizedArray<Number>;
      if constexpr (std::is_same_v<VA, Number2>) {
        /* Vectorized fast access. index must be divisible by the SIMD width */

        std::array<unsigned int, VA::size()> indices;
        for (unsigned int k = 0; k < VA::size(); ++k) {
          AssertIndexRange(js[k], n_locally_relevant_);
          indices[k] = js[k] * n_comp;
        }

        dealii::vectorized_load_and_transpose(
            n_comp, data_, indices.data(), &tensor[0]);

      } else {
        /* Non-vectorized sequential access. */

        AssertIndexRange(*js, n_locally_relevant_);

        for (unsigned int d = 0; d < n_comp; ++d)
          tensor[d] = data_[js[0] * n_comp + d];
      }

      return tensor;
    }

    /**
     * Write a (scalar valued) @p entry to the vector at position by @p i.
     *
     * If the template parameter @a Number2 is a VectorizedArray then
     * the function takes a SIMD vectorized @p tensor as argument instead
     * and updates the values of the @p n_comp component vectors at
     * indices i, i+1, ..., i+n-1 (n being the SIMD width) with the values
     * supplied by @p tensor.
     *
     * @note This function is only available if `n_comp` is equal to 1.
     */
    template <typename Number2 = Number>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    write_entry(const Number2 &entry, const unsigned int i) const
      requires writable
    {
      static_assert(n_comp == 1,
                    "Attempted to write a scalar value into a tensor-valued "
                    "vector entry");

      AssertIndexRange(i, n_locally_relevant_);

      dealii::Tensor<1, n_comp, Number2> tensor;
      tensor[0] = entry;

      write_tensor<Number2>(tensor, i);
    }

    /**
     * Update the values of the @p n_comp component vector at index
     * @p i with the values supplied by @p tensor.
     *
     * If the template parameter @a Number2 is a VectorizedArray then
     * the function takes a SIMD vectorized @p tensor as argument instead
     * and updates the values of the @p n_comp component vectors at
     * indices i, i+1, ..., i+n-1 (n being the SIMD width) with the values
     * supplied by @p tensor.
     *
     * @note @p tensor can be an arbitrary indexable container, such as
     * dealii::Tensor or std::array, that has an `operator[]()` returning a @p
     * Number, and has a type trait `value_type`.
     */
    template <typename Number2 = Number,
              typename Tensor = dealii::Tensor<1, n_comp, Number2>>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    write_tensor(const Tensor &tensor, const unsigned int i) const
      requires writable
    {
      store</*add*/ false, Number2>(tensor, i);
    }

    /**
     * Add a (scalar valued) @p entry to the vector at position @p i.
     * Update the values of the @p n_comp component vector at index @p i
     * by adding the values supplied by @p tensor.
     *
     * If the template parameter @a Number2 is a VectorizedArray then
     * the function takes a SIMD vectorized @p tensor as argument instead
     * and updates the values of the @p n_comp component vectors at
     * indices i, i+1, ..., i+n-1 (n being the SIMD width) with the values
     * supplied by @p tensor.
     *
     * @note This function is only available if `n_comp` is equal to 1.
     */

    template <typename Number2 = Number>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void add_entry(const Number2 &entry,
                                                     const unsigned int i) const
      requires writable
    {
      static_assert(n_comp == 1,
                    "Attempted to write a scalar value into a tensor-valued "
                    "matrix entry");

      AssertIndexRange(i, n_locally_relevant_);

      dealii::Tensor<1, n_comp, Number2> tensor;
      tensor[0] = entry;

      add_tensor<Number2>(tensor, i);
    }

    /**
     * Update the values of the @p n_comp component vector at index @p i
     * by adding the values supplied by @p tensor.
     *
     * If the template parameter @a Number2 is a VectorizedArray then
     * the function takes a SIMD vectorized @p tensor as argument instead
     * and updates the values of the @p n_comp component vectors at
     * indices i, i+1, ..., i+n-1 (n being the SIMD width) with the values
     * supplied by @p tensor.
     *
     * @note @p tensor can be an arbitrary indexable container, such as
     * dealii::Tensor or std::array, that has an `operator[]()` returning a @p
     * Number, and has a type trait `value_type`.
     */
    template <typename Number2 = Number,
              typename Tensor = dealii::Tensor<1, n_comp, Number2>>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    add_tensor(const Tensor &tensor, const unsigned int i) const
      requires writable
    {
      store</*add*/ true, Number2>(tensor, i);
    }

    //@}
    /**
     * @name MPI synchronization
     */
    //@{

    /**
     * MPI synchronization: Zero out all ghost values stored in the vector.
     */
    void zero_out_ghost_values() const
      requires(writable)
    {
      Assert(multi_component_vector_->template is_resident<MemorySpace>(),
             dealii::ExcMessage("The chosen memory space is not resident."));

      multi_component_vector_->template storage<MemorySpace>()
          .zero_out_ghost_values();
    }

    /**
     * MPI synchronization: Import all ghost values from neighboring MPI
     * ranks on the templated memory space.
     */
    void update_ghost_values() const
      requires(writable)
    {
      Assert(multi_component_vector_->template is_resident<MemorySpace>(),
             dealii::ExcMessage("The chosen memory space is not resident."));

      multi_component_vector_->template storage<MemorySpace>()
          .update_ghost_values();
    }

    /**
     * MPI synchronization: Copy the data that has accumulated in the
     * ghost range to the owning processor. This function operates on the
     * templated memory space.
     */
    void compress(dealii::VectorOperation::values operation) const
      requires(writable)
    {
      Assert(multi_component_vector_->template is_resident<MemorySpace>(),
             dealii::ExcMessage("The chosen memory space is not resident."));

      multi_component_vector_->template storage<MemorySpace>().compress(
          operation);
    }

  private:
    //@}
    /**
     * @name Internal fields
     */
    //@{

    using MCV = MultiComponentVector<Number>;
    std::conditional_t<writable, MCV *, const MCV *> multi_component_vector_;

    Number *data_;
    unsigned int n_locally_owned_;
    unsigned int n_locally_relevant_;

    template <typename, int, typename, bool>
    friend class MultiComponentVectorView;

    template <bool add, typename Number2, typename Tensor>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void store(const Tensor &tensor,
                                                 const unsigned int i) const
    {
      static_assert(std::is_same_v<Number2, typename Tensor::value_type>,
                    "type mismatch");

      AssertIndexRange(i, n_locally_relevant_);

      /* Special case of a zero component vector */
      if constexpr (n_comp == 0)
        return;

      using VA = dealii::VectorizedArray<Number>;
      if constexpr (std::is_same_v<VA, Number2>) {
        /* Vectorized fast access. index must be divisible by the SIMD width */

        std::array<unsigned int, VA::size()> indices;
        for (unsigned int k = 0; k < VA::size(); ++k)
          indices[k] = k * n_comp;

        dealii::vectorized_transpose_and_store(
            add, n_comp, &tensor[0], indices.data(), data_ + i * n_comp);

      } else {
        /* Non-vectorized sequential access. */

        for (unsigned int d = 0; d < n_comp; ++d) {
          if constexpr (add)
            data_[i * n_comp + d] += tensor[d];
          else
            data_[i * n_comp + d] = tensor[d];
        }
      }
    }

    //@}
  };


#ifndef DOXYGEN
  /*
   * -------------------------------------------------------------------------
   * Inline function definitions
   * -------------------------------------------------------------------------
   */


  template <typename Number>
  MultiComponentVector<Number>::MultiComponentVector(
      const MultiComponentVector &other)
  {
    *this = other;
  }


  template <typename Number>
  MultiComponentVector<Number>::MultiComponentVector(
      MultiComponentVector &&other) noexcept
  {
    *this = std::move(other);
  }


  template <typename Number>
  void MultiComponentVector<Number>::reinit_with_vector_partitioner(
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          &vector_partitioner,
      const unsigned int n_comp,
      const TransferPolicy transfer_policy)
  {
    n_comp_ = n_comp;

    /*
     * Drop the transfer policy for the duration of the reinit so that a
     * pinned memory space of a previous policy does not interfere:
     */
    this->set_transfer_policy(TransferPolicy::explicit_transfers);

    /* Special case of a zero component vector */
    if (n_comp == 0) {
      /* Drop possibly stale storage from a previous reinit: */
      host_vector_.reinit(0);
      if constexpr (have_separate_memory_spaces)
        default_vector_.reinit(0);

      /* A zero component vector is trivially resident everywhere: */
      this->reset_residency(/*host*/ true, /*default*/ true);
      this->set_transfer_policy(transfer_policy);
      return;
    }

    host_vector_.reinit(vector_partitioner);

    /*
     * The vector is resident on the host memory space only. Device
     * storage is allocated lazily on the first copy_to_memory_space() /
     * move_to_memory_space(); drop possibly stale device storage from a
     * previous reinit:
     */
    if constexpr (have_separate_memory_spaces)
      default_vector_.reinit(0);

    /*
     * If both memory spaces coincide the single host allocation is
     * trivially resident on both of them:
     */
    this->reset_residency(/*host*/ true,
                          /*default*/ !have_separate_memory_spaces);

    this->set_transfer_policy(transfer_policy);
  }


  template <typename Number>
  void MultiComponentVector<Number>::reinit_with_scalar_partitioner(
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          &scalar_partitioner,
      const unsigned int n_comp,
      const TransferPolicy transfer_policy)
  {
    /*
     * A scalar vector (and a zero component vector) uses the scalar
     * partitioner directly:
     */
    const auto vector_partitioner =
        n_comp <= 1 ? scalar_partitioner
                    : create_vector_partitioner(scalar_partitioner, n_comp);

    reinit_with_vector_partitioner(vector_partitioner, n_comp, transfer_policy);
  }


  template <typename Number>
  auto
  MultiComponentVector<Number>::operator=(const MultiComponentVector &other)
      -> MultiComponentVector &
  {
    /* Copy residency state and transfer policy: */
    static_cast<MirroredStorage<MultiComponentVector> &>(*this) = other;
    n_comp_ = other.n_comp_;

    host_vector_ = other.host_vector_;
    if constexpr (have_separate_memory_spaces)
      default_vector_ = other.default_vector_;

    return *this;
  }


  template <typename Number>
  auto
  MultiComponentVector<Number>::operator=(MultiComponentVector &&other) noexcept
      -> MultiComponentVector &
  {
    /* Copy residency state and transfer policy: */
    static_cast<MirroredStorage<MultiComponentVector> &>(*this) = other;
    n_comp_ = other.n_comp_;

    host_vector_ = std::move(other.host_vector_);
    if constexpr (have_separate_memory_spaces)
      default_vector_ = std::move(other.default_vector_);

    return *this;
  }


  template <typename Number>
  template <int n_comp, typename MemorySpace>
  MultiComponentVectorView<Number, n_comp, MemorySpace, true>
  MultiComponentVector<Number>::view()
  {
    Assert(static_cast<unsigned int>(n_comp) == n_comp_,
           dealii::ExcMessage("The number of components of the view does "
                              "not match the number of components of the "
                              "vector."));

    this->template prepare_write_access<MemorySpace>();

    return MultiComponentVectorView<Number, n_comp, MemorySpace, true>(*this);
  }


  template <typename Number>
  template <int n_comp, typename MemorySpace>
  MultiComponentVectorView<Number, n_comp, MemorySpace, false>
  MultiComponentVector<Number>::view() const
  {
    Assert(static_cast<unsigned int>(n_comp) == n_comp_,
           dealii::ExcMessage("The number of components of the view does "
                              "not match the number of components of the "
                              "vector."));

    this->template prepare_read_access<MemorySpace>();

    return MultiComponentVectorView<Number, n_comp, MemorySpace, false>(*this);
  }


  template <typename Number>
  template <typename MemorySpace>
  auto &MultiComponentVector<Number>::deal_ii_vector()
  {
    this->template prepare_write_access<MemorySpace>();
    return storage<MemorySpace>();
  }


  template <typename Number>
  template <typename MemorySpace>
  const auto &MultiComponentVector<Number>::deal_ii_vector() const
  {
    this->template prepare_read_access<MemorySpace>();
    return storage<MemorySpace>();
  }


  template <typename Number>
  template <typename MemorySpace>
  auto &MultiComponentVector<Number>::storage() const
  {
    using HostSpace = dealii::MemorySpace::Host;
    using DefaultSpace = dealii::MemorySpace::Default;
    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    if constexpr (have_separate_memory_spaces &&
                  !std::is_same_v<MemorySpace, HostSpace>)
      return default_vector_;
    else
      return host_vector_;
  }


  template <typename Number>
  template <typename MemorySpace>
  void MultiComponentVector<Number>::allocate_storage() const
  {
    using HostSpace = dealii::MemorySpace::Host;

    /*
     * The partitioner is recovered from the other (still resident)
     * vector:
     */

    if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
      Assert(default_vector_.size() != 0, dealii::ExcNotInitialized());
      host_vector_.reinit(default_vector_.get_partitioner());
    } else {
      Assert(host_vector_.size() != 0, dealii::ExcNotInitialized());
      default_vector_.reinit(host_vector_.get_partitioner());
    }
  }


  template <typename Number>
  template <typename To, typename From>
  void MultiComponentVector<Number>::deep_copy_storage() const
  {
    using HostSpace = dealii::MemorySpace::Host;

    if constexpr (std::is_same_v<To, HostSpace>) {
      host_vector_.import_elements(default_vector_,
                                   dealii::VectorOperation::insert);
    } else {
      default_vector_.import_elements(host_vector_,
                                      dealii::VectorOperation::insert);
    }
  }


  template <typename Number>
  template <typename MemorySpace>
  void MultiComponentVector<Number>::deallocate_storage()
  {
    using HostSpace = dealii::MemorySpace::Host;

    if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
      host_vector_.reinit(0);

    } else {
      default_vector_.reinit(0);
    }
  }

#endif
} // namespace ryujin
