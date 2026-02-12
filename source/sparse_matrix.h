//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "sparsity_pattern.h"

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

namespace ryujin
{
  template <typename Number, int n_components, int simd_length>
  class SparseMatrixView;


  /**
   * A specialized sparse matrix for efficient vectorized SIMD access.
   *
   * In the vectorized row index region [0, n_internal_dofs) we store data
   * as an array-of-struct-of-array type (see the documentation of class
   * SparsityPattern for details). For the non-vectorized row index
   * region [n_internal_dofs, n_locally_relevant_dofs) we store the matrix in
   * CSR format (equivalent to the static dealii::SparsityPattern).
   */
  template <typename Number, int n_components, int simd_length>
  class SparseMatrix
      : public SparseMatrixView<Number, n_components, simd_length>
  {
  public:
    /**
     * Constructor and initialization.
     */
    //@{

    /**
     * Default constructor.
     */
    SparseMatrix();

    /**
     * Constructor taking a SIMD sparsity pattern as an argument.
     */
    SparseMatrix(const SparsityPattern<simd_length> &sparsity);

    /**
     * Reinit function reinitializes the matrix with the given SIMD
     * sparsity pattern.
     */
    void reinit(const SparsityPattern<simd_length> &sparsity);

    //@}
    /**
     * Reading in values from (scalar) matrices.
     */
    //@{

    /**
     * Read in values from a given vector of (scalar) sparse matrices that
     * describe our (vector valued) matrix entries.
     */
    template <typename SparseMatrix2>
    void read_in(const std::array<SparseMatrix2, n_components> &sparse_matrix,
                 bool locally_indexed = true);

    /**
     * Variant of above function for a scalar matrix with n_components == 1.
     */
    template <typename SparseMatrix2>
    void read_in(const SparseMatrix2 &sparse_matrix2,
                 bool locally_indexed = true);

    //@}
    /**
     * MPI synchronization.
     */
    //@{

    void update_ghost_rows_start(const unsigned int communication_channel = 0);

    void update_ghost_rows_finish();

    void update_ghost_rows();

    //@}

  protected:
    /**
     * @name Internal fields, methods, and friends
     */
    //@{

    const SparsityPattern<simd_length> *sparsity = nullptr;

    dealii::AlignedVector<Number> data;
    dealii::AlignedVector<Number> exchange_buffer;
    std::vector<MPI_Request> requests;

    template <typename, int, int>
    friend class SparseMatrixView;

    //@}
  };


  /**
   * This class models a "view" of the sparse matrix that lives in the host
   * or device memory space. It provides a number of methods for reading
   * and writing matrix entries.
   *
   * @note This class is designed to be captured by value in computation
   * loops with access to either the host or device memory space. As such
   * we do not store a reference to the underlying SparsityPattern but
   * rather raw pointers into the corresponding memory. The view is only
   * valid as long as the underlying SparsityPattern object is not
   * modified.
   */
  template <typename Number, int n_components, int simd_length>
  class SparseMatrixView
  {
  public:
    SparseMatrixView() = default;

    SparseMatrixView(
        SparseMatrix<Number, n_components, simd_length> &sparse_matrix);

    void reinit(SparseMatrix<Number, n_components, simd_length> &sparse_matrix);

    /* Get scalar or tensor-valued entry: */

    /**
     * Return the (scalar) entry indexed by @p row and @p
     * position_within_column.
     *
     * @note If the template argument @a Number2
     * is a vetorized array a specialized, faster access will be performed.
     * In this case the index @p row must be within the interval
     * [0, n_internal_dofs) and must be divisible by simd_length.
     *
     * @note This function is only available if `n_components` is equal to 1.
     */
    template <typename Number2 = Number>
    Number2 read_entry(const unsigned int row,
                       const unsigned int position_within_column) const;

    /**
     * Return the tensor-valued entry indexed by @p row and
     * @p position_within_column. This function performs the same operation
     * as read_entry() except that it always returns the entry as a tensor
     * (even if it is effectively a scalar entry).
     *
     * @note If the template argument @a Number2
     * is a vetorized array a specialized, faster access will be performed.
     * In this case the index @p row must be within the interval
     * [0, n_internal_dofs) and must be divisible by simd_length.
     */
    template <typename Number2 = Number,
              typename Tensor = dealii::Tensor<1, n_components, Number2>>
    Tensor read_tensor(const unsigned int row,
                       const unsigned int position_within_column) const;

    /* Get transposed scalar or tensor-valued entry: */

    /**
     * Return the transposed (sclar) entry indexed by @p row and
     * @p position_within_column.
     *
     * @note If the template argument @a Number2
     * is a vetorized array a specialized, faster access will be performed.
     * In this case the index @p row must be within the interval
     * [0, n_internal_dofs) and must be divisible by simd_length.
     *
     * @note This function is only available if `n_components` is equal to 1.
     */
    template <typename Number2 = Number>
    Number2
    read_transposed_entry(const unsigned int row,
                          const unsigned int position_within_column) const;

    /**
     * Return the transposed tensor-valued entry indexed by @p row and
     * @a position_within_column. This function performs the same operation
     * as read_entry() except that it always returns the entry as a tensor
     * (even if it is effectively a scalar entry).
     *
     * @note If the template argument @a Number2
     * is a vetorized array a specialized, faster access will be performed.
     * In this case the index @p row must be within the interval
     * [0, n_internal_dofs) and must be divisible by simd_length.
     */
    template <typename Number2 = Number,
              typename Tensor = dealii::Tensor<1, n_components, Number2>>
    Tensor
    read_transposed_tensor(const unsigned int row,
                           const unsigned int position_within_column) const;

    /* Write scalar or tensor entry: */

    /**
     * Write a (scalar valued) @p entry to the matrix indexed by @p row
     * and @p position_within_column.
     *
     * @note If the template argument @a Number2
     * is a vetorized array a specialized, faster access will be performed.
     * In this case the index @p row must be within the interval
     * [0, n_internal_dofs) and must be divisible by simd_length.
     *
     * @note This function is only available if `n_components` is equal to 1.
     */
    template <typename Number2 = Number>
    void write_entry(const Number2 entry,
                     const unsigned int row,
                     const unsigned int position_within_column,
                     const bool do_streaming_store = false);

    /**
     * Write a tensor-valued @p entry to the matrix indexed by @p row
     * and @p position_within_column.
     *
     * @note If the template argument @a Number2
     * is a vetorized array a specialized, faster access will be performed.
     * In this case the index @p row must be within the interval
     * [0, n_internal_dofs) and must be divisible by simd_length.
     */
    template <typename Number2 = Number,
              typename Tensor = dealii::Tensor<1, n_components, Number2>>
    void write_tensor(const Tensor &tensor,
                      const unsigned int row,
                      const unsigned int position_within_column,
                      const bool do_streaming_store = false);

  private:
    const SparsityPatternView<simd_length> *sparsity = nullptr;

    using HostSpace = dealii::MemorySpace::Host::kokkos_space;
    Kokkos::View<Number *, HostSpace> data_view;
  };


#ifndef DOXYGEN
  /*
   * -------------------------------------------------------------------------
   * Inline function definitions
   * -------------------------------------------------------------------------
   */


  template <typename Number, int n_components, int simd_length>
  SparseMatrixView<Number, n_components, simd_length>::SparseMatrixView(
      SparseMatrix<Number, n_components, simd_length> &sparse_matrix)
  {
    reinit(sparse_matrix);
  }

  template <typename Number, int n_components, int simd_length>
  void SparseMatrixView<Number, n_components, simd_length>::reinit(
      SparseMatrix<Number, n_components, simd_length> &sparse_matrix)
  {
    using unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;

    sparsity = static_cast<const SparsityPatternView<simd_length> *>(
        sparse_matrix.sparsity);

    data_view = Kokkos::View<Number *, HostSpace, unmanaged>(
        sparse_matrix.data.data(), sparse_matrix.data.size());
  }


  template <typename Number, int n_components, int simd_length>
  template <typename Number2>
  DEAL_II_ALWAYS_INLINE inline Number2
  SparseMatrixView<Number, n_components, simd_length>::read_entry(
      const unsigned int row, const unsigned int position_within_column) const
  {
    static_assert(
        n_components == 1,
        "Attempted to write a scalar value into a tensor-valued matrix entry");

    const auto result = read_tensor<Number2>(row, position_within_column);
    return result[0];
  }


  template <typename Number, int n_components, int simd_length>
  template <typename Number2, typename Tensor>
  DEAL_II_ALWAYS_INLINE inline Tensor
  SparseMatrixView<Number, n_components, simd_length>::read_tensor(
      const unsigned int row, const unsigned int position_within_column) const
  {
    using VA = dealii::VectorizedArray<Number>;

    Assert(sparsity != nullptr, dealii::ExcNotInitialized());
    AssertIndexRange(row, sparsity->n_rows());
    AssertIndexRange(position_within_column, sparsity->row_length(row));

    static_assert(std::is_same_v<Number2, typename Tensor::value_type>,
                  "type mismatch");

    Tensor result;

    if constexpr (std::is_same_v<Number, Number2>) {
      /*
       * Non-vectorized slow access. Supports all row indices in
       * [0,n_owned)
       */
      for (unsigned int d = 0; d < n_components; ++d) {
        const auto offset = sparsity->template offset<n_components>(
            row, position_within_column, d);
        result[d] = data_view(offset);
      }

    } else if constexpr (std::is_same_v<VA, Number2>) {
      /*
       * Vectorized fast access. Indices must be in the range
       * [0,n_internal), index must be divisible by simd_length
       */

      Assert(row < sparsity->n_internal_dofs(),
             dealii::ExcMessage(
                 "Vectorized access only possible in vectorized part"));
      Assert(row % simd_length == 0,
             dealii::ExcMessage(
                 "Access only supported for rows at the SIMD granularity"));

      const Number *load_pos = data_view.data();
      load_pos += sparsity->template offset_internal<n_components>(
          row, position_within_column);

      for (unsigned int d = 0; d < n_components; ++d)
        result[d].load(load_pos + d * simd_length);

    } else {
      /* not implemented */
      __builtin_trap();
    }

    return result;
  }


  template <typename Number, int n_components, int simd_length>
  template <typename Number2>
  DEAL_II_ALWAYS_INLINE inline Number2
  SparseMatrixView<Number, n_components, simd_length>::read_transposed_entry(
      const unsigned int row, const unsigned int position_within_column) const
  {
    static_assert(
        n_components == 1,
        "Attempted to write a scalar value into a tensor-valued matrix entry");

    const auto result =
        read_transposed_tensor<Number2>(row, position_within_column);
    return result[0];
  }


  template <typename Number, int n_components, int simd_length>
  template <typename Number2, typename Tensor>
  DEAL_II_ALWAYS_INLINE inline Tensor
  SparseMatrixView<Number, n_components, simd_length>::read_transposed_tensor(
      const unsigned int row, const unsigned int position_within_column) const
  {
    using VA = dealii::VectorizedArray<Number>;

    Assert(sparsity != nullptr, dealii::ExcNotInitialized());
    AssertIndexRange(row, sparsity->n_rows());
    AssertIndexRange(position_within_column, sparsity->row_length(row));

    static_assert(std::is_same_v<Number2, typename Tensor::value_type>,
                  "type mismatch");

    dealii::Tensor<1, n_components, Number2> result;

    if constexpr (std::is_same_v<Number, Number2>) {
      /*
       * Non-vectorized slow access. Supports all row indices in
       * [0,n_owned)
       */
      for (unsigned int d = 0; d < n_components; ++d) {
        const auto offset = sparsity->template transposed_offset<n_components>(
            row, position_within_column, d);
        result[d] = data_view(offset);
      }

    } else if constexpr (std::is_same_v<VA, Number2> && (n_components == 1)) {
      /*
       * Vectorized fast access. Indices must be in the range
       * [0,n_internal), index must be divisible by simd_length
       */

      Assert(row < sparsity->n_internal_dofs(),
             dealii::ExcMessage(
                 "Vectorized access only possible in vectorized part"));
      Assert(row % simd_length == 0,
             dealii::ExcMessage(
                 "Access only supported for rows at the SIMD granularity"));

      const auto offsets = sparsity->template transposed_offset_internal<1>(
          row, position_within_column);
      result[0].gather(data_view.data(), offsets);

    } else {
      /* not implemented */
      Assert(false,
             dealii::ExcMessage("Vectorized transposed access to multiple "
                                "components is not implemented."));
      __builtin_trap();
    }

    return result;
  }


  template <typename Number, int n_components, int simd_length>
  template <typename Number2>
  DEAL_II_ALWAYS_INLINE inline void
  SparseMatrixView<Number, n_components, simd_length>::write_entry(
      const Number2 entry,
      const unsigned int row,
      const unsigned int position_within_column,
      const bool do_streaming_store)
  {
    static_assert(
        n_components == 1,
        "Attempted to write a scalar value into a tensor-valued matrix entry");

    Assert(sparsity != nullptr, dealii::ExcNotInitialized());
    AssertIndexRange(row, sparsity->n_rows());
    AssertIndexRange(position_within_column, sparsity->row_length(row));

    dealii::Tensor<1, n_components, Number2> tensor;
    tensor[0] = entry;

    write_tensor<Number2>(
        tensor, row, position_within_column, do_streaming_store);
  }


  template <typename Number, int n_components, int simd_length>
  template <typename Number2, typename Tensor>
  DEAL_II_ALWAYS_INLINE inline void
  SparseMatrixView<Number, n_components, simd_length>::write_tensor(
      const Tensor &tensor,
      const unsigned int row,
      const unsigned int position_within_column,
      const bool do_streaming_store)
  {
    using VA = dealii::VectorizedArray<Number>;

    Assert(sparsity != nullptr, dealii::ExcNotInitialized());
    AssertIndexRange(row, sparsity->n_rows());
    AssertIndexRange(position_within_column, sparsity->row_length(row));

    if constexpr (std::is_same_v<Number, Number2>) {
      /*
       * Non-vectorized slow access. Supports all row indices in
       * [0,n_owned)
       */
      for (unsigned int d = 0; d < n_components; ++d) {
        const auto offset = sparsity->template offset<n_components>(
            row, position_within_column, d);
        data_view[offset] = tensor[d];
      }

    } else if constexpr (std::is_same_v<VA, Number2>) {
      /*
       * Vectorized fast access. Indices must be in the range
       * [0,n_internal), index must be divisible by simd_length
       */

      Assert(row < sparsity->n_internal_dofs(),
             dealii::ExcMessage(
                 "Vectorized access only possible in vectorized part"));
      Assert(row % simd_length == 0,
             dealii::ExcMessage(
                 "Access only supported for rows at the SIMD granularity"));

      Number *store_pos = data_view.data();
      store_pos += sparsity->template offset_internal<n_components>(
          row, position_within_column);

      if (do_streaming_store)
        for (unsigned int d = 0; d < n_components; ++d)
          tensor[d].streaming_store(store_pos + d * simd_length);
      else
        for (unsigned int d = 0; d < n_components; ++d)
          tensor[d].store(store_pos + d * simd_length);

    } else {
      /* not implemented */
      __builtin_trap();
    }
  }

#endif
} // namespace ryujin
