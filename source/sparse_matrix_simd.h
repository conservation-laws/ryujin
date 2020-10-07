//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef SPARSE_MATRIX_SIMD
#define SPARSE_MATRIX_SIMD

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
    SparsityPatternSIMD();

    SparsityPatternSIMD(
        const unsigned int n_internal_dofs,
        const dealii::DynamicSparsityPattern &sparsity,
        const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
            &partitioner);


    /**
     *
     */
    void reinit(const unsigned int n_internal_dofs,
                const dealii::DynamicSparsityPattern &sparsity,
                const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
                    &partitioner);

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

    dealii::AlignedVector<std::size_t> indices_to_be_sent;
    std::vector<std::pair<unsigned int, unsigned int>> send_targets;
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

    Number get_entry(const unsigned int row,
                     const unsigned int position_within_column) const;

    dealii::Tensor<1, n_components, Number>
    get_tensor(const unsigned int row,
               const unsigned int position_within_column) const;

    VectorizedArray
    get_vectorized_entry(const unsigned int row,
                         const unsigned int position_within_column) const;

    dealii::Tensor<1, n_components, VectorizedArray>
    get_vectorized_tensor(const unsigned int row,
                          const unsigned int position_within_column) const;

    /* Get transposed scalar or tensor-valued entry: */

    Number
    get_transposed_entry(const unsigned int row,
                         const unsigned int position_within_column) const;

    dealii::Tensor<1, n_components, Number>
    get_transposed_tensor(const unsigned int row,
                          const unsigned int position_within_column) const;

    VectorizedArray get_vectorized_transposed_entry(
        const unsigned int row,
        const unsigned int position_within_column) const;

    /* FIXME: Implement get_vectorized_transposed_tensor */

    /* Write scalar or tensor entry: */

    void write_entry(const Number entry,
                     const unsigned int row,
                     const unsigned int position_within_column);

    void write_tensor(const dealii::Tensor<1, n_components, Number> &entry,
                      const unsigned int row,
                      const unsigned int position_within_column);

    void write_vectorized_entry(const VectorizedArray entry,
                                const unsigned int row,
                                const unsigned int position_within_column,
                                const bool do_streaming_store = false);

    void write_vectorized_tensor(
        const dealii::Tensor<1, n_components, VectorizedArray> &entry,
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
  DEAL_II_ALWAYS_INLINE inline Number
  SparseMatrixSIMD<Number, n_components, simd_length>::get_entry(
      const unsigned int row, const unsigned int position_within_column) const
  {
    return get_tensor(row, position_within_column)[0];
  }


  template <typename Number, int n_components, int simd_length>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, n_components, Number>
  SparseMatrixSIMD<Number, n_components, simd_length>::get_tensor(
      const unsigned int row, const unsigned int position_within_column) const
  {
    Assert(sparsity != nullptr, dealii::ExcNotInitialized());

    AssertIndexRange(row, sparsity->row_starts.size() - 1);
    AssertIndexRange(position_within_column, sparsity->row_length(row));

    dealii::Tensor<1, n_components, Number> result;
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
        result[d] = data[(sparsity->row_starts[row] + position_within_column) *
                             n_components +
                         d];
    }

    return result;
  }


  template <typename Number, int n_components, int simd_length>
  DEAL_II_ALWAYS_INLINE inline dealii::VectorizedArray<Number, simd_length>
  SparseMatrixSIMD<Number, n_components, simd_length>::get_vectorized_entry(
      const unsigned int row, const unsigned int position_within_column) const
  {
    return get_vectorized_tensor(row, position_within_column)[0];
  }


  template <typename Number, int n_components, int simd_length>
  DEAL_II_ALWAYS_INLINE inline auto
  SparseMatrixSIMD<Number, n_components, simd_length>::get_vectorized_tensor(
      const unsigned int row, const unsigned int position_within_column) const
      -> dealii::Tensor<1, n_components, VectorizedArray>
  {
    Assert(sparsity != nullptr, dealii::ExcNotInitialized());

    AssertIndexRange(row, sparsity->row_starts.size() - 1);
    AssertIndexRange(position_within_column, sparsity->row_length(row));
    Assert(row < sparsity->n_internal_dofs,
           dealii::ExcMessage(
               "Vectorized access only possible in vectorized part"));
    Assert(row % simd_length == 0,
           dealii::ExcMessage(
               "Access only supported for rows at the SIMD granularity"));

    dealii::
        Tensor<1, n_components, dealii::VectorizedArray<Number, simd_length>>
            result;
    const Number *load_pos =
        data.data() + (sparsity->row_starts[row / simd_length] +
                       position_within_column * simd_length) *
                          n_components;
    for (unsigned int d = 0; d < n_components; ++d)
      result[d].load(load_pos + d * simd_length);
    return result;
  }


  template <typename Number, int n_components, int simd_length>
  DEAL_II_ALWAYS_INLINE inline Number
  SparseMatrixSIMD<Number, n_components, simd_length>::get_transposed_entry(
      const unsigned int row, const unsigned int position_within_column) const
  {
    return get_transposed_tensor(row, position_within_column)[0];
  }


  template <typename Number, int n_components, int simd_length>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, n_components, Number>
  SparseMatrixSIMD<Number, n_components, simd_length>::get_transposed_tensor(
      const unsigned int row, const unsigned int position_within_column) const
  {
    Assert(sparsity != nullptr, dealii::ExcNotInitialized());

    AssertIndexRange(row, sparsity->row_starts.size() - 1);
    AssertIndexRange(position_within_column, sparsity->row_length(row));

    dealii::Tensor<1, n_components, Number> result;
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
            sparsity
                ->column_indices[sparsity->row_starts[simd_row] + simd_offset +
                                 position_within_column * simd_length];
        if (col < sparsity->n_internal_dofs)
          for (unsigned int d = 0; d < n_components; ++d)
            result[d] = data[index / simd_length * simd_length * n_components +
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
            result[d] = data[index / simd_length * simd_length * n_components +
                             simd_length * d + index % simd_length];
        else
          for (unsigned int d = 0; d < n_components; ++d)
            result[d] = data[index * n_components + d];
      } else
        result[0] = data[index];
    }
    return result;
  }


  template <typename Number, int n_components, int simd_length>
  DEAL_II_ALWAYS_INLINE inline dealii::VectorizedArray<Number, simd_length>
  SparseMatrixSIMD<Number, n_components, simd_length>::
      get_vectorized_transposed_entry(
          const unsigned int row,
          const unsigned int position_within_column) const
  {
    Assert(sparsity != nullptr, dealii::ExcNotInitialized());

    AssertIndexRange(row, sparsity->row_starts.size() - 1);
    AssertIndexRange(position_within_column, sparsity->row_length(row));
    Assert(row < sparsity->n_internal_dofs,
           dealii::ExcMessage(
               "Vectorized access only possible in vectorized part"));
    Assert(row % simd_length == 0,
           dealii::ExcMessage(
               "Access only supported for rows at the SIMD granularity"));

    const unsigned int offset = sparsity->row_starts[row / simd_length] +
                                position_within_column * simd_length;
    dealii::VectorizedArray<Number, simd_length> result;
    result.gather(data.data(), sparsity->indices_transposed.data() + offset);
    return result;
  }


  template <typename Number, int n_components, int simd_length>
  DEAL_II_ALWAYS_INLINE inline void
  SparseMatrixSIMD<Number, n_components, simd_length>::write_entry(
      const Number entry,
      const unsigned int row,
      const unsigned int position_within_column)
  {
    dealii::Tensor<1, n_components, Number> result;
    result[0] = entry;

    write_tensor(result, row, position_within_column);
  }


  template <typename Number, int n_components, int simd_length>
  DEAL_II_ALWAYS_INLINE inline void
  SparseMatrixSIMD<Number, n_components, simd_length>::write_tensor(
      const dealii::Tensor<1, n_components, Number> &entry,
      const unsigned int row,
      const unsigned int position_within_column)
  {
    Assert(sparsity != nullptr, dealii::ExcNotInitialized());

    AssertIndexRange(row, sparsity->row_starts.size() - 1);
    AssertIndexRange(position_within_column, sparsity->row_length(row));

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
  }


  template <typename Number, int n_components, int simd_length>
  DEAL_II_ALWAYS_INLINE inline void
  SparseMatrixSIMD<Number, n_components, simd_length>::write_vectorized_entry(
      const dealii::VectorizedArray<Number, simd_length> entry,
      const unsigned int row,
      const unsigned int position_within_column,
      const bool do_streaming_store)
  {
    dealii::Tensor<1, n_components, VectorizedArray> tensor;
    tensor[0] = entry;

    write_vectorized_tensor(
        tensor, row, position_within_column, do_streaming_store);
  }


  template <typename Number, int n_components, int simd_length>
  DEAL_II_ALWAYS_INLINE inline void
  SparseMatrixSIMD<Number, n_components, simd_length>::write_vectorized_tensor(
      const dealii::Tensor<1, n_components, VectorizedArray> &entry,
      const unsigned int row,
      const unsigned int position_within_column,
      const bool do_streaming_store)
  {
    Assert(sparsity != nullptr, dealii::ExcNotInitialized());

    AssertIndexRange(row, sparsity->row_starts.size() - 1);
    AssertIndexRange(position_within_column, sparsity->row_length(row));
    Assert(row < sparsity->n_internal_dofs,
           dealii::ExcMessage(
               "Vectorized access only possible in vectorized part"));
    Assert(row % simd_length == 0,
           dealii::ExcMessage(
               "Access only supported for rows at the SIMD granularity"));
    Number *store_pos = data.data() + (sparsity->row_starts[row / simd_length] +
                                       position_within_column * simd_length) *
                                          n_components;
    if (do_streaming_store)
      for (unsigned int d = 0; d < n_components; ++d)
        entry[d].streaming_store(store_pos + d * simd_length);
    else
      for (unsigned int d = 0; d < n_components; ++d)
        entry[d].store(store_pos + d * simd_length);
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

    const std::size_t n_indices = sparsity->indices_to_be_sent.size();
    exchange_buffer.resize_fast(n_components * n_indices);

    requests.resize(sparsity->receive_targets.size() +
                    sparsity->send_targets.size());
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

    RYUJIN_PARALLEL_REGION_BEGIN

    RYUJIN_OMP_FOR
    for (std::size_t c = 0; c < n_indices; ++c)
      for (unsigned int comp = 0; comp < n_components; ++comp)
        exchange_buffer[n_components * c + comp] =
            data[n_components * sparsity->indices_to_be_sent[c] +
                 comp ];

    RYUJIN_PARALLEL_REGION_END

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

#endif
