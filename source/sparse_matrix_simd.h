//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include "openmp.h"
#include "simd.h"

namespace ryujin
{
  template <typename Number,
            int n_components = 1,
            int simd_length = dealii::VectorizedArray<Number>::size()>
  class SparseMatrixSIMD;

  /**
   * A specialized sparsity pattern for efficient vectorized SIMD access.
   *
   * In the vectorized row index region [0, n_internal_dofs) we store data
   * as an array-of-struct-of-array type as follows:
   *
   *  - At the innermost 'array' level, we group data from simd_length rows
   *    contiguously in memory, using a given column block as determined by
   *    the sparsity pattern.
   *
   *  - Next come the different components in case we have a
   *    multi-component matrix, i.e., the 'struct' level groups the
   *    components next to the inner array of row data.
   *
   *  - Finally, the outer array aligns the different components in a CSR
   *    format, i.e., row-by-row (or row-chunk-per-row-chunk) and along
   *    columns, following the sparsity pattern.
   *
   * For the non-vectorized row index region [n_internal_dofs,
   * n_locally_relevant_dofs) we store the matrix in CSR format (equivalent
   * to the static dealii::SparsityPattern).
   */
  template <int simd_length>
  class SparsityPatternSIMD
  {
  public:
    /**
     * Default constructor.
     */
    SparsityPatternSIMD();

    /**
     * Constructor taking a sparsity pattern template, an MPI partitioner
     * and the number of (regular) internal dofs as an argument. The
     * constructor calls SparsityPatternSIMD::reinit() internally.
     */
    SparsityPatternSIMD(
        const unsigned int n_internal_dofs,
        const dealii::DynamicSparsityPattern &sparsity,
        const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
            &partitioner);


    /**
     * Reinit function that reinitializes the SIMD sparsity pattern for a
     * given sparsity pattern template, an MPI partitioner and the number
     * of (regular) internal dofs.
     */
    void reinit(const unsigned int n_internal_dofs,
                const dealii::DynamicSparsityPattern &sparsity,
                const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
                    &partitioner);

    /**
     * Return the "stride size" of a given row index. The function returns
     * simd_length for all indices in the range [0, n_internal_dofs) and 1
     * otherwise.
     */
    unsigned int stride_of_row(const unsigned int row) const;

    const unsigned int *columns(const unsigned int row) const;

    unsigned int row_length(const unsigned int row) const;

    unsigned int n_rows() const;

    std::size_t n_nonzero_elements() const;

  private:
    unsigned int n_internal_dofs;
    unsigned int n_locally_owned_dofs;
    std::shared_ptr<const dealii::Utilities::MPI::Partitioner> partitioner;

    dealii::AlignedVector<std::size_t> row_starts;
    dealii::AlignedVector<unsigned int> column_indices;
    dealii::AlignedVector<unsigned int> indices_transposed;

    /**
     * Array listing all (locally owned) entries as a pair {row,
     * position_within_column}, potentially duplicated, and arranged
     * consecutively by send targets.
     */
    std::vector<std::pair<unsigned int, unsigned int>> entries_to_be_sent;

    /**
     * All send targets stored as a pair consisting of an MPI rank (first
     * entry) and a corresponding index range into entries_to_be_sent given
     * by the half open range [send_targets[p-1].second, send_targets[p])
     */
    std::vector<std::pair<unsigned int, unsigned int>> send_targets;

    /**
     * All receive targets stored as a pair consisting of an MPI rank (first
     * entry) and a corresponding index range into the (serial) data()
     * array given by the half open range [receive_targets[p-1].second,
     * receive_targets[p])
     */
    std::vector<std::pair<unsigned int, unsigned int>> receive_targets;

    MPI_Comm mpi_communicator;

    template <typename, int, int>
    friend class SparseMatrixSIMD;
  };


  /**
   * A specialized sparse matrix for efficient vectorized SIMD access.
   *
   * In the vectorized row index region [0, n_internal_dofs) we store data
   * as an array-of-struct-of-array type (see the documentation of class
   * SparsityPatternSIMD for details). For the non-vectorized row index
   * region [n_internal_dofs, n_locally_relevant_dofs) we store the matrix in
   * CSR format (equivalent to the static dealii::SparsityPattern).
   */
  template <typename Number, int n_components, int simd_length>
  class SparseMatrixSIMD
  {
  public:
    SparseMatrixSIMD();

    SparseMatrixSIMD(const SparsityPatternSIMD<simd_length> &sparsity);

    void reinit(const SparsityPatternSIMD<simd_length> &sparsity);

    template <typename SparseMatrix>
    void read_in(const std::array<SparseMatrix, n_components> &sparse_matrix,
                 bool locally_indexed = true);

    template <typename SparseMatrix>
    void read_in(const SparseMatrix &sparse_matrix,
                 bool locally_indexed = true);

    using VectorizedArray = dealii::VectorizedArray<Number, simd_length>;

    /* Get scalar or tensor-valued entry: */

    /**
     * return the (scalar) entry indexed by @p row and
     * @p position_within_column.
     *
     * @note If the template argument @a Number2
     * is a vetorized array a specialized, faster access will be performed.
     * In this case the index @p row must be within the interval
     * [0, n_internal_dofs) and must be divisible by simd_length.
     */
    template <typename Number2 = Number>
    Number2 get_entry(const unsigned int row,
                      const unsigned int position_within_column) const;

    /**
     * return the tensor-valued entry indexed by @p row and
     * @p position_within_column.
     *
     * @note If the template argument @a Number2
     * is a vetorized array a specialized, faster access will be performed.
     * In this case the index @p row must be within the interval
     * [0, n_internal_dofs) and must be divisible by simd_length.
     */
    template <typename Number2 = Number>
    dealii::Tensor<1, n_components, Number2>
    get_tensor(const unsigned int row,
               const unsigned int position_within_column) const;

    /* Get transposed scalar or tensor-valued entry: */

    /**
     * return the transposed (scalar) entry indexed by @p row and
     * @p position_within_column.
     *
     * @note If the template argument @a Number2
     * is a vetorized array a specialized, faster access will be performed.
     * In this case the index @p row must be within the interval
     * [0, n_internal_dofs) and must be divisible by simd_length.
     */
    template <typename Number2 = Number>
    Number2
    get_transposed_entry(const unsigned int row,
                         const unsigned int position_within_column) const;

    /**
     * return the transposed tensor-valued entry indexed by @p row and
     * @a position_within_column.
     *
     * @note If the template argument @a Number2
     * is a vetorized array a specialized, faster access will be performed.
     * In this case the index @p row must be within the interval
     * [0, n_internal_dofs) and must be divisible by simd_length.
     */
    template <typename Number2 = Number>
    dealii::Tensor<1, n_components, Number2>
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
    template <typename Number2 = Number>
    void write_tensor(const dealii::Tensor<1, n_components, Number2> &entry,
                      const unsigned int row,
                      const unsigned int position_within_column,
                      const bool do_streaming_store = false);

    /* Synchronize over MPI ranks: */

    void update_ghost_rows_start(const unsigned int communication_channel = 0);

    void update_ghost_rows_finish();

    void update_ghost_rows();

  private:
    const SparsityPatternSIMD<simd_length> *sparsity;
    dealii::AlignedVector<Number> data;
    dealii::AlignedVector<Number> exchange_buffer;
    std::vector<MPI_Request> requests;
  };

  /*
   * Inline function  definitions:
   */


  template <int simd_length>
  DEAL_II_ALWAYS_INLINE inline unsigned int
  SparsityPatternSIMD<simd_length>::stride_of_row(const unsigned int row) const
  {
    AssertIndexRange(row, row_starts.size());

    if (row < n_internal_dofs)
      return simd_length;
    else
      return 1;
  }


  template <int simd_length>
  DEAL_II_ALWAYS_INLINE inline const unsigned int *
  SparsityPatternSIMD<simd_length>::columns(const unsigned int row) const
  {
    AssertIndexRange(row, row_starts.size() - 1);

    if (row < n_internal_dofs)
      return column_indices.data() + row_starts[row / simd_length] +
             row % simd_length;
    else
      return column_indices.data() + row_starts[row];
  }


  template <int simd_length>
  DEAL_II_ALWAYS_INLINE inline unsigned int
  SparsityPatternSIMD<simd_length>::row_length(const unsigned int row) const
  {
    AssertIndexRange(row, row_starts.size() - 1);

    if (row < n_internal_dofs) {
      const unsigned int simd_row = row / simd_length;
      return (row_starts[simd_row + 1] - row_starts[simd_row]) / simd_length;
    } else {
      return row_starts[row + 1] - row_starts[row];
    }
  }


  template <int simd_length>
  DEAL_II_ALWAYS_INLINE inline unsigned int
  SparsityPatternSIMD<simd_length>::n_rows() const
  {
    return row_starts.size() - 1;
  }


  template <int simd_length>
  DEAL_II_ALWAYS_INLINE inline std::size_t
  SparsityPatternSIMD<simd_length>::n_nonzero_elements() const
  {
    Assert(row_starts.size() > 0, dealii::ExcNotInitialized());

    return row_starts.back();
  }


  template <typename Number, int n_components, int simd_length>
  template <typename Number2>
  DEAL_II_ALWAYS_INLINE inline Number2
  SparseMatrixSIMD<Number, n_components, simd_length>::get_entry(
      const unsigned int row, const unsigned int position_within_column) const
  {
    return get_tensor<Number2>(row, position_within_column)[0];
  }


  template <typename Number, int n_components, int simd_length>
  template <typename Number2>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, n_components, Number2>
  SparseMatrixSIMD<Number, n_components, simd_length>::get_tensor(
      const unsigned int row, const unsigned int position_within_column) const
  {
    Assert(sparsity != nullptr, dealii::ExcNotInitialized());
    AssertIndexRange(row, sparsity->row_starts.size() - 1);
    AssertIndexRange(position_within_column, sparsity->row_length(row));

    dealii::Tensor<1, n_components, Number2> result;

    if constexpr (std::is_same<Number, Number2>::value) {
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

    } else if constexpr (std::is_same<VectorizedArray, Number2>::value) {
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
  DEAL_II_ALWAYS_INLINE inline Number2
  SparseMatrixSIMD<Number, n_components, simd_length>::get_transposed_entry(
      const unsigned int row, const unsigned int position_within_column) const
  {
    return get_transposed_tensor<Number2>(row, position_within_column)[0];
  }


  template <typename Number, int n_components, int simd_length>
  template <typename Number2>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, n_components, Number2>
  SparseMatrixSIMD<Number, n_components, simd_length>::get_transposed_tensor(
      const unsigned int row, const unsigned int position_within_column) const
  {
    Assert(sparsity != nullptr, dealii::ExcNotInitialized());
    AssertIndexRange(row, sparsity->row_starts.size() - 1);
    AssertIndexRange(position_within_column, sparsity->row_length(row));

    dealii::Tensor<1, n_components, Number2> result;

    if constexpr (std::is_same<Number, Number2>::value) {
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

    } else if constexpr (std::is_same<VectorizedArray, Number2>::value &&
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
  SparseMatrixSIMD<Number, n_components, simd_length>::write_entry(
      const Number2 entry,
      const unsigned int row,
      const unsigned int position_within_column,
      const bool do_streaming_store)
  {
    Assert(sparsity != nullptr, dealii::ExcNotInitialized());
    AssertIndexRange(row, sparsity->row_starts.size() - 1);
    AssertIndexRange(position_within_column, sparsity->row_length(row));

    dealii::Tensor<1, n_components, Number2> result;
    result[0] = entry;

    write_tensor<Number2>(
        result, row, position_within_column, do_streaming_store);
  }


  template <typename Number, int n_components, int simd_length>
  template <typename Number2>
  DEAL_II_ALWAYS_INLINE inline void
  SparseMatrixSIMD<Number, n_components, simd_length>::write_tensor(
      const dealii::Tensor<1, n_components, Number2> &entry,
      const unsigned int row,
      const unsigned int position_within_column,
      const bool do_streaming_store)
  {
    Assert(sparsity != nullptr, dealii::ExcNotInitialized());
    AssertIndexRange(row, sparsity->row_starts.size() - 1);
    AssertIndexRange(position_within_column, sparsity->row_length(row));

    if constexpr (std::is_same<Number, Number2>::value) {
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
               d * simd_length + simd_offset] = entry[d];
      } else {
        // go through standard part
        for (unsigned int d = 0; d < n_components; ++d)
          data[(sparsity->row_starts[row] + position_within_column) *
                   n_components +
               d] = entry[d];
      }

    } else if constexpr (std::is_same<VectorizedArray, Number2>::value) {
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
          entry[d].streaming_store(store_pos + d * simd_length);
      else
        for (unsigned int d = 0; d < n_components; ++d)
          entry[d].store(store_pos + d * simd_length);

    } else {
      /* not implemented */
      __builtin_trap();
    }
  }


  template <typename Number, int n_components, int simd_length>
  inline void
  SparseMatrixSIMD<Number, n_components, simd_length>::update_ghost_rows_start(
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
  inline void SparseMatrixSIMD<Number, n_components, simd_length>::
      update_ghost_rows_finish()
  {
#ifdef DEAL_II_WITH_MPI
    const int ierr =
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    AssertThrowMPI(ierr);
#endif
  }


  template <typename Number, int n_components, int simd_length>
  inline void
  SparseMatrixSIMD<Number, n_components, simd_length>::update_ghost_rows()
  {
    update_ghost_rows_start();
    update_ghost_rows_finish();
  }

} // namespace ryujin
