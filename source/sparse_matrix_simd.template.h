//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include "sparse_matrix_simd.h"

#include <deal.II/base/vectorization.h>
#include <deal.II/lac/sparse_matrix.h>

namespace ryujin
{

  template <int simd_length>
  SparsityPatternSIMD<simd_length>::SparsityPatternSIMD()
      : n_internal_dofs(0)
      , row_starts(1)
      , mpi_communicator(MPI_COMM_SELF)
  {
  }


  template <int simd_length>
  SparsityPatternSIMD<simd_length>::SparsityPatternSIMD(
      const unsigned int n_internal_dofs,
      const dealii::DynamicSparsityPattern &sparsity,
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          &partitioner)
      : n_internal_dofs(0)
      , mpi_communicator(MPI_COMM_SELF)
  {
    reinit(n_internal_dofs, sparsity, partitioner);
  }


  template <int simd_length>
  void SparsityPatternSIMD<simd_length>::reinit(
      const unsigned int n_internal_dofs,
      const dealii::DynamicSparsityPattern &dsp,
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          &partitioner)
  {
    this->mpi_communicator = partitioner->get_mpi_communicator();

    this->n_internal_dofs = n_internal_dofs;
    this->n_locally_owned_dofs = partitioner->locally_owned_size();
    this->partitioner = partitioner;

    const auto n_locally_relevant_dofs =
        partitioner->locally_owned_size() + partitioner->n_ghost_indices();

    /*
     * First, create a static sparsity pattern (in local indexing), where
     * the only off-processor rows are the ones for which locally owned rows
     * request the transpose entries. This will be the one we finally
     * compute on.
     */

    dealii::DynamicSparsityPattern dsp_minimal(n_locally_relevant_dofs,
                                               n_locally_relevant_dofs);
    for (unsigned int i = 0; i < n_locally_owned_dofs; ++i) {
      const auto global_row = partitioner->local_to_global(i);
      for (auto it = dsp.begin(global_row); it != dsp.end(global_row); ++it) {
        const auto global_column = it->column();
        const auto j = partitioner->global_to_local(global_column);
        dsp_minimal.add(i, j);
        if (j >= n_locally_owned_dofs) {
          Assert(j < n_locally_relevant_dofs, dealii::ExcInternalError());
          dsp_minimal.add(j, i);
        }
      }
    }

    dealii::SparsityPattern sparsity;
    sparsity.copy_from(dsp_minimal);

    Assert(n_internal_dofs <= sparsity.n_rows(), dealii::ExcInternalError());
    Assert(n_internal_dofs % simd_length == 0, dealii::ExcInternalError());
    Assert(n_internal_dofs <= n_locally_owned_dofs, dealii::ExcInternalError());
    Assert(n_locally_owned_dofs <= sparsity.n_rows(),
           dealii::ExcInternalError());

    row_starts.resize_fast(sparsity.n_rows() + 1);
    column_indices.resize_fast(sparsity.n_nonzero_elements());
    indices_transposed.resize_fast(sparsity.n_nonzero_elements());
    AssertThrow(sparsity.n_nonzero_elements() <
                    std::numeric_limits<unsigned int>::max(),
                dealii::ExcMessage("Transposed indices only support up to 4 "
                                   "billion matrix entries per MPI rank. Try to"
                                   " split into smaller problems with MPI"));

    /* Vectorized part: */

    row_starts[0] = 0;

    unsigned int *col_ptr = column_indices.data();
    unsigned int *transposed_ptr = indices_transposed.data();

    for (unsigned int i = 0; i < n_internal_dofs; i += simd_length) {
      auto jts = generate_iterators<simd_length>(
          [&](auto k) { return sparsity.begin(i + k); });

      for (; jts[0] != sparsity.end(i); increment_iterators(jts))
        for (unsigned int k = 0; k < simd_length; ++k) {
          const unsigned int column = jts[k]->column();
          *col_ptr++ = column;
          const std::size_t position = sparsity(column, i + k);
          if (column < n_internal_dofs) {
            const unsigned int my_row_length = sparsity.row_length(column);
            const std::size_t position_diag = sparsity(column, column);
            const std::size_t pos_within_row = position - position_diag;
            const unsigned int simd_offset = column % simd_length;
            *transposed_ptr++ = position - simd_offset * my_row_length -
                                pos_within_row + simd_offset +
                                pos_within_row * simd_length;
          } else
            *transposed_ptr++ = position;
        }

      row_starts[i / simd_length + 1] = col_ptr - column_indices.data();
    }

    /* Rest: */

    row_starts[n_internal_dofs] = row_starts[n_internal_dofs / simd_length];

    for (unsigned int i = n_internal_dofs; i < sparsity.n_rows(); ++i) {
      for (auto j = sparsity.begin(i); j != sparsity.end(i); ++j) {
        const unsigned int column = j->column();
        *col_ptr++ = column;
        const std::size_t position = sparsity(column, i);
        if (column < n_internal_dofs) {
          const unsigned int my_row_length = sparsity.row_length(column);
          const std::size_t position_diag = sparsity(column, column);
          const std::size_t pos_within_row = position - position_diag;
          const unsigned int simd_offset = column % simd_length;
          *transposed_ptr++ = position - simd_offset * my_row_length -
                              pos_within_row + simd_offset +
                              pos_within_row * simd_length;
        } else
          *transposed_ptr++ = position;
      }
      row_starts[i + 1] = col_ptr - column_indices.data();
    }

    Assert(col_ptr == column_indices.end(), dealii::ExcInternalError());

    /* Compute the data exchange pattern: */

    if (sparsity.n_rows() > n_locally_owned_dofs) {

      /*
       * Set up receive targets.
       *
       * We receive our (reduced) ghost rows from MPI ranks in the ghost
       * range of the (scalar) partitioner. We receive the entire (reduced)
       * ghost row from that MPI rank; we can thus simply query the
       * sparsity pattern how many data points we receive from each MPI
       * rank.
       */

      const auto &ghost_targets = partitioner->ghost_targets();

      receive_targets.resize(ghost_targets.size());

      for (unsigned int p = 0; p < receive_targets.size(); ++p) {
        receive_targets[p].first = ghost_targets[p].first;
      }

      const auto gt_begin = ghost_targets.begin();
      auto gt_ptr = ghost_targets.begin();
      std::size_t index = 0; /* index into ghost range of sparsity pattern */
      unsigned int row_count = 0;

      for (unsigned int i = n_locally_owned_dofs; i < sparsity.n_rows(); ++i) {
        index += sparsity.row_length(i);
        ++row_count;
        if (row_count == gt_ptr->second) {
          receive_targets[gt_ptr - gt_begin].second = index;
          row_count = 0; /* reset row count and move on to new rank */
          ++gt_ptr;
        }
      }

      Assert(gt_ptr == partitioner->ghost_targets().end(),
             dealii::ExcInternalError());

      /*
       * Collect indices to be sent.
       *
       * These consist of the diagonal, as well as the part of columns of
       * the given range. Note that this assumes that the sparsity pattern
       * only contains those entries in ghosted rows which have a
       * corresponding transpose entry in the owned rows; which is the case
       * for our minimized sparsity pattern.
       */

      std::vector<unsigned int> ghost_ranges(ghost_targets.size() + 1);
      ghost_ranges[0] = n_locally_owned_dofs;
      for (unsigned int p = 0; p < receive_targets.size(); ++p) {
        ghost_ranges[p + 1] = ghost_ranges[p] + ghost_targets[p].second;
      }

      std::vector<unsigned int> import_indices_part;
      for (auto i : partitioner->import_indices())
        for (unsigned int j = i.first; j < i.second; ++j)
          import_indices_part.push_back(j);

      AssertDimension(import_indices_part.size(),
                      partitioner->n_import_indices());

      const auto &import_targets = partitioner->import_targets();
      entries_to_be_sent.clear();
      send_targets.resize(import_targets.size());
      auto idx = import_indices_part.begin();

      /*
       * Index p iterates over import_targets() and index p_match iterates
       * over ghost_ranges. such that
       *   import_targets()[p].first == ghost_rangs[p_match].first
       */
      unsigned int p_match = 0;
      for (unsigned int p = 0; p < import_targets.size(); ++p) {

        /*
         * Match up the rank index between receive and import targets. If
         * we do not find a match, which can happen for locally refined
         * meshes, then we set p_match equal to receive_targets.size().
         *
         * When trying to match the next processor index we consequently
         * have to reset p_match to 0 again. This assumes that processor
         * indices are sorted in the receive_targets and ghost_targets
         * vectors.
         */
        p_match = (p_match == receive_targets.size() ? 0 : p_match);
        while (p_match < receive_targets.size() &&
               receive_targets[p_match].first != import_targets[p].first)
          p_match++;

        for (unsigned int c = 0; c < import_targets[p].second; ++c, ++idx) {
          /*
           * Continue if we do not have a match. Note that we need to enter
           * and continue this loop till the end in order to advance idx
           * correctly.
           */
          if (p_match == receive_targets.size())
            continue;

          const unsigned int row = *idx;

          entries_to_be_sent.emplace_back(row, 0);
          for (auto jt = ++sparsity.begin(row); jt != sparsity.end(row); ++jt) {
            if (jt->column() >= ghost_ranges[p_match] &&
                jt->column() < ghost_ranges[p_match + 1]) {
              const unsigned int position_within_column =
                  jt - sparsity.begin(row);
              entries_to_be_sent.emplace_back(row, position_within_column);
            }
          }
        }

        send_targets[p].first = partitioner->import_targets()[p].first;
        send_targets[p].second = entries_to_be_sent.size();
      }
    }
  }


  template <typename Number, int n_components, int simd_length>
  SparseMatrixSIMD<Number, n_components, simd_length>::SparseMatrixSIMD()
      : sparsity(nullptr)
  {
  }


  template <typename Number, int n_components, int simd_length>
  SparseMatrixSIMD<Number, n_components, simd_length>::SparseMatrixSIMD(
      const SparsityPatternSIMD<simd_length> &sparsity)
      : sparsity(&sparsity)
  {
    data.resize(sparsity.n_nonzero_elements() * n_components);
  }


  template <typename Number, int n_components, int simd_length>
  void SparseMatrixSIMD<Number, n_components, simd_length>::reinit(
      const SparsityPatternSIMD<simd_length> &sparsity)
  {
    this->sparsity = &sparsity;
    data.resize(sparsity.n_nonzero_elements() * n_components);
  }


  template <typename Number, int n_components, int simd_length>
  template <typename SparseMatrix>
  void SparseMatrixSIMD<Number, n_components, simd_length>::read_in(
      const std::array<SparseMatrix, n_components> &sparse_matrix,
      bool locally_indexed /*= true*/)
  {
    RYUJIN_PARALLEL_REGION_BEGIN

    /*
     * We use the indirect (and slow) access via operator()(i, j) into the
     * sparse matrix we are copying from. This allows for significantly
     * increased flexibility with respect to the sparsity pattern used in
     * the sparse_matrix object.
     */

    RYUJIN_OMP_FOR
    for (unsigned int i = 0; i < sparsity->n_internal_dofs; i += simd_length) {

      const unsigned int row_length = sparsity->row_length(i);

      const unsigned int *js = sparsity->columns(i);
      for (unsigned int col_idx = 0; col_idx < row_length;
           ++col_idx, js += simd_length) {

        dealii::Tensor<1, n_components, VectorizedArray> temp;
        for (unsigned int k = 0; k < simd_length; ++k)
          for (unsigned int d = 0; d < n_components; ++d)
            if (locally_indexed)
              temp[d][k] = sparse_matrix[d](i + k, js[k]);
            else
              temp[d][k] = sparse_matrix[d].el(
                  sparsity->partitioner->local_to_global(i + k),
                  sparsity->partitioner->local_to_global(js[k]));

        write_entry(temp, i, col_idx, true);
      }
    }

    RYUJIN_OMP_FOR
    for (unsigned int i = sparsity->n_internal_dofs;
         i < sparsity->n_locally_owned_dofs;
         ++i) {
      const unsigned int row_length = sparsity->row_length(i);
      const unsigned int *js = sparsity->columns(i);
      for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx, ++js) {

        dealii::Tensor<1, n_components, Number> temp;
        for (unsigned int d = 0; d < n_components; ++d)
          if (locally_indexed)
            temp[d] = sparse_matrix[d](i, js[0]);
          else
            temp[d] = sparse_matrix[d].el(
                sparsity->partitioner->local_to_global(i),
                sparsity->partitioner->local_to_global(js[0]));
        write_entry(temp, i, col_idx);
      }
    }

    RYUJIN_PARALLEL_REGION_END
  }


  template <typename Number, int n_components, int simd_length>
  template <typename SparseMatrix>
  void SparseMatrixSIMD<Number, n_components, simd_length>::read_in(
      const SparseMatrix &sparse_matrix, bool locally_indexed /*= true*/)
  {
    RYUJIN_PARALLEL_REGION_BEGIN

    /*
     * We use the indirect (and slow) access via operator()(i, j) into the
     * sparse matrix we are copying from. This allows for significantly
     * increased flexibility with respect to the sparsity pattern used in
     * the sparse_matrix object.
     */

    RYUJIN_OMP_FOR
    for (unsigned int i = 0; i < sparsity->n_internal_dofs; i += simd_length) {

      const unsigned int row_length = sparsity->row_length(i);

      const unsigned int *js = sparsity->columns(i);
      for (unsigned int col_idx = 0; col_idx < row_length;
           ++col_idx, js += simd_length) {

        VectorizedArray temp = {};
        for (unsigned int k = 0; k < simd_length; ++k)
          if (locally_indexed)
            temp[k] = sparse_matrix(i + k, js[k]);
          else
            temp[k] =
                sparse_matrix.el(sparsity->partitioner->local_to_global(i + k),
                                 sparsity->partitioner->local_to_global(js[k]));

        write_entry(temp, i, col_idx, true);
      }
    }

    RYUJIN_OMP_FOR
    for (unsigned int i = sparsity->n_internal_dofs;
         i < sparsity->n_locally_owned_dofs;
         ++i) {

      const unsigned int row_length = sparsity->row_length(i);
      const unsigned int *js = sparsity->columns(i);
      for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx, ++js) {

        const Number temp =
            locally_indexed
                ? sparse_matrix(i, js[0])
                : sparse_matrix.el(
                      sparsity->partitioner->local_to_global(i),
                      sparsity->partitioner->local_to_global(js[0]));
        write_entry(temp, i, col_idx);
      }
    }

    RYUJIN_PARALLEL_REGION_END
  }

} // namespace ryujin
