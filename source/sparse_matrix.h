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
  {
  public:
    SparseMatrix();

    SparseMatrix(const SparsityPattern<simd_length> &sparsity);

    void reinit(const SparsityPattern<simd_length> &sparsity);

    template <typename SparseMatrix2>
    void read_in(const std::array<SparseMatrix2, n_components> &sparse_matrix,
                 bool locally_indexed = true);

    template <typename SparseMatrix2>
    void read_in(const SparseMatrix2 &sparse_matrix,
                 bool locally_indexed = true);

    using VectorizedArray = dealii::VectorizedArray<Number, simd_length>;

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
    Tensor get_tensor(const unsigned int row,
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
    get_transposed_entry(const unsigned int row,
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
    get_transposed_tensor(const unsigned int row,
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

    /* Synchronize over MPI ranks: */

    void update_ghost_rows_start(const unsigned int communication_channel = 0);

    void update_ghost_rows_finish();

    void update_ghost_rows();

  protected:
    const SparsityPattern<simd_length> *sparsity;
    dealii::AlignedVector<Number> data;
    dealii::AlignedVector<Number> exchange_buffer;
    std::vector<MPI_Request> requests;
  };


#ifndef DOXYGEN
    /*
     * -------------------------------------------------------------------------
     * Inline function definitions
     * -------------------------------------------------------------------------
     */

  template <typename Number, int n_components, int simd_length>
  template <typename Number2>
  DEAL_II_ALWAYS_INLINE inline auto
  SparseMatrix<Number, n_components, simd_length>::read_entry(
      const unsigned int row, const unsigned int position_within_column) const
      -> Number2
  {
    static_assert(
        n_components == 1,
        "Attempted to write a scalar value into a tensor-valued matrix entry");

    const auto result = get_tensor<Number2>(row, position_within_column);
    return result[0];
  }


  template <typename Number, int n_components, int simd_length>
  template <typename Number2, typename Tensor>
  DEAL_II_ALWAYS_INLINE inline Tensor
  SparseMatrix<Number, n_components, simd_length>::get_tensor(
      const unsigned int row, const unsigned int position_within_column) const
  {
    Assert(sparsity != nullptr, dealii::ExcNotInitialized());
    AssertIndexRange(row, sparsity->row_starts.size() - 1);
    AssertIndexRange(position_within_column, sparsity->row_length(row));

    static_assert(std::is_same_v<Number2, typename Tensor::value_type>,
                  "type mismatch");

    Tensor result;

    if constexpr (std::is_same_v<Number, Number2>) {
      /*
       * Non-vectorized slow access. Supports all row indices in
       * [0,n_owned)
       */
      if (row < sparsity->n_internal_dofs) {
        // go through vectorized part
        const unsigned int simd_row = row / simd_length;
        const unsigned int simd_offset = row % simd_length;
        for (unsigned int d = 0; d < n_components; ++d)
          result[d] = data[(sparsity->row_starts[simd_row] +
                            position_within_column * simd_length) *
                               n_components +
                           d * simd_length + simd_offset];
      } else {
        // go through standard part
        for (unsigned int d = 0; d < n_components; ++d)
          result[d] =
              data[(sparsity->row_starts[row] + position_within_column) *
                       n_components +
                   d];
      }

    } else if constexpr (std::is_same_v<VectorizedArray, Number2>) {
      /*
       * Vectorized fast access. Indices must be in the range
       * [0,n_internal), index must be divisible by simd_length
       */

      Assert(row < sparsity->n_internal_dofs,
             dealii::ExcMessage(
                 "Vectorized access only possible in vectorized part"));
      Assert(row % simd_length == 0,
             dealii::ExcMessage(
                 "Access only supported for rows at the SIMD granularity"));

      const Number *load_pos =
          data.data() + (sparsity->row_starts[row / simd_length] +
                         position_within_column * simd_length) *
                            n_components;

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
  DEAL_II_ALWAYS_INLINE inline auto
  SparseMatrix<Number, n_components, simd_length>::get_transposed_entry(
      const unsigned int row, const unsigned int position_within_column) const
      -> Number2
  {
    static_assert(
        n_components == 1,
        "Attempted to write a scalar value into a tensor-valued matrix entry");

    const auto result =
        get_transposed_tensor<Number2>(row, position_within_column);
    return result[0];
  }


  template <typename Number, int n_components, int simd_length>
  template <typename Number2, typename Tensor>
  DEAL_II_ALWAYS_INLINE inline Tensor
  SparseMatrix<Number, n_components, simd_length>::get_transposed_tensor(
      const unsigned int row, const unsigned int position_within_column) const
  {
    Assert(sparsity != nullptr, dealii::ExcNotInitialized());
    AssertIndexRange(row, sparsity->row_starts.size() - 1);
    AssertIndexRange(position_within_column, sparsity->row_length(row));

    static_assert(std::is_same_v<Number2, typename Tensor::value_type>,
                  "type mismatch");

    Tensor result;

    if constexpr (std::is_same_v<Number, Number2>) {
      /*
       * Non-vectorized slow access. Supports all row indices in
       * [0,n_owned)
       */

      if (row < sparsity->n_internal_dofs) {
        // go through vectorized part
        const unsigned int simd_row = row / simd_length;
        const unsigned int simd_offset = row % simd_length;
        const std::size_t index =
            sparsity->indices_transposed[sparsity->row_starts[simd_row] +
                                         simd_offset +
                                         position_within_column * simd_length];
        if (n_components > 1) {
          const unsigned int col =
              sparsity->column_indices[sparsity->row_starts[simd_row] +
                                       simd_offset +
                                       position_within_column * simd_length];
          if (col < sparsity->n_internal_dofs)
            for (unsigned int d = 0; d < n_components; ++d)
              result[d] =
                  data[index / simd_length * simd_length * n_components +
                       simd_length * d + index % simd_length];
          else
            for (unsigned int d = 0; d < n_components; ++d)
              result[d] = data[index * n_components + d];
        } else
          result[0] = data[index];
      } else {
        // go through standard part
        const std::size_t index =
            sparsity->indices_transposed[sparsity->row_starts[row] +
                                         position_within_column];
        if (n_components > 1) {
          const unsigned int col =
              sparsity->column_indices[sparsity->row_starts[row] +
                                       position_within_column];
          if (col < sparsity->n_internal_dofs)
            for (unsigned int d = 0; d < n_components; ++d)
              result[d] =
                  data[index / simd_length * simd_length * n_components +
                       simd_length * d + index % simd_length];
          else
            for (unsigned int d = 0; d < n_components; ++d)
              result[d] = data[index * n_components + d];
        } else
          result[0] = data[index];
      }

    } else if constexpr (std::is_same_v<VectorizedArray, Number2> &&
                         (n_components == 1)) {
      /*
       * Vectorized fast access. Indices must be in the range
       * [0,n_internal), index must be divisible by simd_length
       */

      Assert(row < sparsity->n_internal_dofs,
             dealii::ExcMessage(
                 "Vectorized access only possible in vectorized part"));
      Assert(row % simd_length == 0,
             dealii::ExcMessage(
                 "Access only supported for rows at the SIMD granularity"));

      const unsigned int offset = sparsity->row_starts[row / simd_length] +
                                  position_within_column * simd_length;
      result[0].gather(data.data(),
                       sparsity->indices_transposed.data() + offset);

    } else {
      /* not implemented */
      __builtin_trap();
    }

    return result;
  }


  template <typename Number, int n_components, int simd_length>
  template <typename Number2>
  DEAL_II_ALWAYS_INLINE inline void
  SparseMatrix<Number, n_components, simd_length>::write_entry(
      const Number2 entry,
      const unsigned int row,
      const unsigned int position_within_column,
      const bool do_streaming_store)
  {
    static_assert(
        n_components == 1,
        "Attempted to write a scalar value into a tensor-valued matrix entry");

    Assert(sparsity != nullptr, dealii::ExcNotInitialized());
    AssertIndexRange(row, sparsity->row_starts.size() - 1);
    AssertIndexRange(position_within_column, sparsity->row_length(row));

    dealii::Tensor<1, n_components, Number2> tensor;
    tensor[0] = entry;

    write_tensor<Number2>(
        tensor, row, position_within_column, do_streaming_store);
  }


  template <typename Number, int n_components, int simd_length>
  template <typename Number2, typename Tensor>
  DEAL_II_ALWAYS_INLINE inline void
  SparseMatrix<Number, n_components, simd_length>::write_tensor(
      const Tensor &tensor,
      const unsigned int row,
      const unsigned int position_within_column,
      const bool do_streaming_store)
  {
    Assert(sparsity != nullptr, dealii::ExcNotInitialized());
    AssertIndexRange(row, sparsity->row_starts.size() - 1);
    AssertIndexRange(position_within_column, sparsity->row_length(row));

    static_assert(std::is_same_v<Number2, typename Tensor::value_type>,
                  "type mismatch");

    if constexpr (std::is_same_v<Number, Number2>) {
      /*
       * Non-vectorized slow access. Supports all row indices in
       * [0,n_owned)
       */

      if (row < sparsity->n_internal_dofs) {
        // go through vectorized part
        const unsigned int simd_row = row / simd_length;
        const unsigned int simd_offset = row % simd_length;
        for (unsigned int d = 0; d < n_components; ++d)
          data[(sparsity->row_starts[simd_row] +
                position_within_column * simd_length) *
                   n_components +
               d * simd_length + simd_offset] = tensor[d];
      } else {
        // go through standard part
        for (unsigned int d = 0; d < n_components; ++d)
          data[(sparsity->row_starts[row] + position_within_column) *
                   n_components +
               d] = tensor[d];
      }

    } else if constexpr (std::is_same_v<VectorizedArray, Number2>) {
      /*
       * Vectorized fast access. Indices must be in the range
       * [0,n_internal), index must be divisible by simd_length
       */

      Assert(row < sparsity->n_internal_dofs,
             dealii::ExcMessage(
                 "Vectorized access only possible in vectorized part"));
      Assert(row % simd_length == 0,
             dealii::ExcMessage(
                 "Access only supported for rows at the SIMD granularity"));

      Number *store_pos =
          data.data() + (sparsity->row_starts[row / simd_length] +
                         position_within_column * simd_length) *
                            n_components;
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


  template <typename Number, int n_components, int simd_length>
  inline void
  SparseMatrix<Number, n_components, simd_length>::update_ghost_rows_start(
      const unsigned int communication_channel)
  {
#ifdef DEAL_II_WITH_MPI
    AssertIndexRange(communication_channel, 200);

    const unsigned int mpi_tag =
        dealii::Utilities::MPI::internal::Tags::partitioner_export_start +
        communication_channel;
    Assert(mpi_tag <=
               dealii::Utilities::MPI::internal::Tags::partitioner_export_end,
           dealii::ExcInternalError());

    const std::size_t n_indices = sparsity->entries_to_be_sent.size();
    exchange_buffer.resize_fast(n_components * n_indices);

    requests.resize(sparsity->receive_targets.size() +
                    sparsity->send_targets.size());

    /*
     * Set up MPI receive requests. We will always receive data for indices
     * in the range [n_locally_owned_, n_locally_relevant_), thus the DATA
     * is stored in non-vectorized CSR format.
     */

    {
      const auto &targets = sparsity->receive_targets;
      for (unsigned int p = 0; p < targets.size(); ++p) {
        const int ierr = MPI_Irecv(
            data.data() +
                n_components *
                    (sparsity->row_starts[sparsity->n_locally_owned_dofs] +
                     (p == 0 ? 0 : targets[p - 1].second)),
            (targets[p].second - (p == 0 ? 0 : targets[p - 1].second)) *
                n_components * sizeof(Number),
            MPI_BYTE,
            targets[p].first,
            mpi_tag,
            sparsity->mpi_communicator,
            &requests[p]);
        AssertThrowMPI(ierr);
      }
    }

    /*
     * Copy all entries that we plan to send over to the exchange buffer.
     * Here, we have to be careful with indices falling into the "locally
     * internal" range that are stored in an array-of-struct-of-array type.
     */

    for (std::size_t c = 0; c < n_indices; ++c) {

      const auto &[row, position_within_column] =
          sparsity->entries_to_be_sent[c];

      Assert(row < sparsity->n_locally_owned_dofs, dealii::ExcInternalError());

      if (row < sparsity->n_internal_dofs) {
        // go through vectorized part
        const unsigned int simd_row = row / simd_length;
        const unsigned int simd_offset = row % simd_length;
        for (unsigned int d = 0; d < n_components; ++d)
          exchange_buffer[n_components * c + d] =
              data[(sparsity->row_starts[simd_row] +
                    position_within_column * simd_length) *
                       n_components +
                   d * simd_length + simd_offset];
      } else {
        // go through standard part
        for (unsigned int d = 0; d < n_components; ++d)
          exchange_buffer[n_components * c + d] =
              data[(sparsity->row_starts[row] + position_within_column) *
                       n_components +
                   d];
      }
    }

    /*
     * Set up MPI send requests. We have copied everything we intend to
     * send to the exchange_buffer compatible with the CSR storage format
     * of the receiving MPI rank.
     */

    {
      const auto &targets = sparsity->send_targets;
      for (unsigned int p = 0; p < targets.size(); ++p) {
        const int ierr = MPI_Isend(
            exchange_buffer.data() +
                n_components * (p == 0 ? 0 : targets[p - 1].second),
            (targets[p].second - (p == 0 ? 0 : targets[p - 1].second)) *
                n_components * sizeof(Number),
            MPI_BYTE,
            targets[p].first,
            mpi_tag,
            sparsity->mpi_communicator,
            &requests[p + sparsity->receive_targets.size()]);
        AssertThrowMPI(ierr);
      }
    }
#endif
  }


  template <typename Number, int n_components, int simd_length>
  inline void
  SparseMatrix<Number, n_components, simd_length>::update_ghost_rows_finish()
  {
#ifdef DEAL_II_WITH_MPI
    const int ierr =
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    AssertThrowMPI(ierr);
#endif
  }


  template <typename Number, int n_components, int simd_length>
  inline void
  SparseMatrix<Number, n_components, simd_length>::update_ghost_rows()
  {
    update_ghost_rows_start();
    update_ghost_rows_finish();
  }

#endif
} // namespace ryujin
