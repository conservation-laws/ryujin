//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2025 by the ryujin authors
//

#pragma once

#include "loop.h"
#include "simd.h"
#include "sparse_matrix.h"

#include <deal.II/base/vectorization.h>
#include <deal.II/lac/sparse_matrix.h>

namespace ryujin
{
  template <typename Number, int n_components, int simd_length>
  SparseMatrix<Number, n_components, simd_length>::SparseMatrix()
      : sparsity(nullptr)
  {
  }


  template <typename Number, int n_components, int simd_length>
  SparseMatrix<Number, n_components, simd_length>::SparseMatrix(
      const SparsityPattern<simd_length> &sparsity)
  {
    reinit(sparsity);
  }


  template <typename Number, int n_components, int simd_length>
  void SparseMatrix<Number, n_components, simd_length>::reinit(
      const SparsityPattern<simd_length> &sparsity)
  {
    this->sparsity = &sparsity;
    data.resize(sparsity.n_nonzero_elements() * n_components);

    /* reinitialize the view: */
    SparseMatrixView<Number, n_components, simd_length>::reinit(*this);
  }


  template <typename Number, int n_components, int simd_length>
  template <typename SparseMatrix2>
  void SparseMatrix<Number, n_components, simd_length>::read_in(
      const std::array<SparseMatrix2, n_components> &sparse_matrix,
      bool locally_indexed /*= true*/)
  {
    /*
     * We use the indirect (and slow) access via operator()(i, j) into the
     * sparse matrix we are copying from. This allows for significantly
     * increased flexibility with respect to the sparsity pattern used in
     * the sparse_matrix object.
     */

    const auto body = [&](auto sentinel, unsigned int i) {
      using T = decltype(sentinel);
      constexpr unsigned int stride_size = get_stride_size<T>;
      static_assert(stride_size == 1 || stride_size == simd_length);

      const unsigned int row_length = sparsity->row_length(i);
      const unsigned int *js = sparsity->columns(i);

      for (unsigned int col_idx = 0; col_idx < row_length;
           ++col_idx, js += stride_size) {

        dealii::Tensor<1, n_components, T> temp;

        using VA = dealii::VectorizedArray<Number, simd_length>;
        if constexpr (std::is_same_v<T, VA>) {
          /* Special access for VectorizedArray: */
          for (unsigned int k = 0; k < simd_length; ++k)
            for (unsigned int d = 0; d < n_components; ++d)
              if (locally_indexed)
                temp[d][k] = sparse_matrix[d](i + k, js[k]);
              else
                temp[d][k] = sparse_matrix[d].el(
                    sparsity->partitioner_->local_to_global(i + k),
                    sparsity->partitioner_->local_to_global(js[k]));

          this->template write_tensor<T>(temp, i, col_idx, true);

        } else {
          for (unsigned int d = 0; d < n_components; ++d)
            if (locally_indexed)
              temp[d] = sparse_matrix[d](i, js[0]);
            else
              temp[d] = sparse_matrix[d].el(
                  sparsity->partitioner_->local_to_global(i),
                  sparsity->partitioner_->local_to_global(js[0]));
          this->template write_tensor<T>(temp, i, col_idx);
        }
      }
    };

    cpu_simd_loop<Number>("sparse_matrix_read_in",
                          body,
                          0,
                          sparsity->n_internal_dofs(),
                          sparsity->n_locally_owned_dofs());
  }


  template <typename Number, int n_components, int simd_length>
  template <typename SparseMatrix2>
  void SparseMatrix<Number, n_components, simd_length>::read_in(
      const SparseMatrix2 &sparse_matrix, bool locally_indexed /*= true*/)
  {
    /*
     * We use the indirect (and slow) access via operator()(i, j) into the
     * sparse matrix we are copying from. This allows for significantly
     * increased flexibility with respect to the sparsity pattern used in
     * the sparse_matrix object.
     */

    const auto body = [&](auto sentinel, unsigned int i) {
      using T = decltype(sentinel);
      constexpr unsigned int stride_size = get_stride_size<T>;
      static_assert(stride_size == 1 || stride_size == simd_length);

      const unsigned int row_length = sparsity->row_length(i);
      const unsigned int *js = sparsity->columns(i);

      for (unsigned int col_idx = 0; col_idx < row_length;
           ++col_idx, js += stride_size) {

        auto temp = T{};

        using VA = dealii::VectorizedArray<Number, simd_length>;
        if constexpr (std::is_same_v<T, VA>) {
          for (unsigned int k = 0; k < simd_length; ++k)
            if (locally_indexed)
              temp[k] = sparse_matrix(i + k, js[k]);
            else
              temp[k] = sparse_matrix.el(
                  sparsity->partitioner_->local_to_global(i + k),
                  sparsity->partitioner_->local_to_global(js[k]));

          this->template write_entry<T>(temp, i, col_idx, true);

        } else {
          temp = locally_indexed
                     ? sparse_matrix(i, js[0])
                     : sparse_matrix.el(
                           sparsity->partitioner_->local_to_global(i),
                           sparsity->partitioner_->local_to_global(js[0]));
          this->template write_entry<T>(temp, i, col_idx);
        }
      }
    };

    cpu_simd_loop<Number>("sparse_matrix_read_in",
                          body,
                          0,
                          sparsity->n_internal_dofs(),
                          sparsity->n_locally_owned_dofs());
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

    const std::size_t n_indices = sparsity->entries_to_be_sent_.size();
    exchange_buffer.resize_fast(n_components * n_indices);

    requests.resize(sparsity->receive_targets_.size() +
                    sparsity->send_targets_.size());

    /*
     * Set up MPI receive requests. We will always receive data for indices
     * in the range [n_locally_owned_, n_locally_relevant_), thus the DATA
     * is stored in non-vectorized CSR format.
     */

    {
      const auto &targets = sparsity->receive_targets_;
      for (unsigned int p = 0; p < targets.size(); ++p) {
        const int ierr = MPI_Irecv(
            data.data() +
                n_components *
                    (sparsity->row_starts_[sparsity->n_locally_owned_dofs_] +
                     (p == 0 ? 0 : targets[p - 1].second)),
            (targets[p].second - (p == 0 ? 0 : targets[p - 1].second)) *
                n_components * sizeof(Number),
            MPI_BYTE,
            targets[p].first,
            mpi_tag,
            sparsity->mpi_communicator_,
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
          sparsity->entries_to_be_sent_[c];

      Assert(row < sparsity->n_locally_owned_dofs_, dealii::ExcInternalError());

      if (row < sparsity->n_internal_dofs_) {
        // go through vectorized part
        const unsigned int simd_row = row / simd_length;
        const unsigned int simd_offset = row % simd_length;
        for (unsigned int d = 0; d < n_components; ++d)
          exchange_buffer[n_components * c + d] =
              data[(sparsity->row_starts_[simd_row] +
                    position_within_column * simd_length) *
                       n_components +
                   d * simd_length + simd_offset];
      } else {
        // go through standard part
        for (unsigned int d = 0; d < n_components; ++d)
          exchange_buffer[n_components * c + d] =
              data[(sparsity->row_starts_[row] + position_within_column) *
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
      const auto &targets = sparsity->send_targets_;
      for (unsigned int p = 0; p < targets.size(); ++p) {
        const int ierr = MPI_Isend(
            exchange_buffer.data() +
                n_components * (p == 0 ? 0 : targets[p - 1].second),
            (targets[p].second - (p == 0 ? 0 : targets[p - 1].second)) *
                n_components * sizeof(Number),
            MPI_BYTE,
            targets[p].first,
            mpi_tag,
            sparsity->mpi_communicator_,
            &requests[p + sparsity->receive_targets_.size()]);
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

} // namespace ryujin
