//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/gpu.h>
#include <ryujin/base/simd.h>
#include <ryujin/linear_algebra/sparsity_pattern.h>

#include <deal.II/base/parallel.h>
#include <deal.II/base/vectorization.h>

#include <algorithm>

namespace ryujin
{
  template <int warp_size>
  SparsityPattern<warp_size>::SparsityPattern()
      : n_internal_dofs_(0)
  {
  }


  template <int warp_size>
  SparsityPattern<warp_size>::SparsityPattern(
      const unsigned int n_internal_dofs,
      const DynamicSparsityPattern &pattern,
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          &partitioner,
      bool symmetrize_ghost_range,
      const TransferPolicy transfer_policy)
  {
    reinit(n_internal_dofs,
           pattern,
           partitioner,
           symmetrize_ghost_range,
           transfer_policy);
  }


  template <int warp_size>
  void SparsityPattern<warp_size>::reinit(
      const unsigned int n_internal_dofs,
      const DynamicSparsityPattern &pattern,
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          &partitioner,
      const bool symmetrize_ghost_range,
      const TransferPolicy transfer_policy)
  {
    RYUJIN_DEBUG_ANNOUNCE();

    /*
     * Drop the transfer policy for the duration of the reinit so that a
     * pinned memory space of a previous policy does not interfere:
     */
    this->set_transfer_policy(TransferPolicy::explicit_transfers);

    this->n_internal_dofs_ = n_internal_dofs;
    this->n_locally_owned_dofs_ = partitioner->locally_owned_size();
    this->partitioner_ = partitioner;

    const auto n_locally_relevant_dofs =
        partitioner->locally_owned_size() + partitioner->n_ghost_indices();

    /*
     * First, create the rows in local indexing. We translate the rows of
     * locally owned degrees of freedom in parallel. Symmetrizing adds
     * entries to the ghost rows only, which we therefore set up
     * afterwards.
     */

    std::vector<std::vector<unsigned int>> rows(n_locally_relevant_dofs);

    const auto translate_row = [&](const unsigned int i) {
      const auto global_row = partitioner->local_to_global(i);
      for (const auto column : pattern.columns(global_row))
        rows[i].push_back(partitioner->global_to_local(column));
    };

    dealii::parallel::apply_to_subranges(
        0u,
        n_locally_owned_dofs_,
        [&](const unsigned int begin, const unsigned int end) {
          for (unsigned int i = begin; i < end; ++i)
            translate_row(i);
        },
        4096);

    for (auto i = n_locally_owned_dofs_; i < n_locally_relevant_dofs; ++i)
      translate_row(i);

    if (symmetrize_ghost_range) {
      for (unsigned int i = 0; i < n_locally_owned_dofs_; ++i)
        for (const auto j : rows[i])
          if (j >= n_locally_owned_dofs_)
            rows[j].push_back(i);
    }

    /*
     * Sort every row and store the diagonal entry first. Every row has a
     * diagonal entry:
     */

    dealii::parallel::apply_to_subranges(
        0u,
        n_locally_relevant_dofs,
        [&](const unsigned int begin, const unsigned int end) {
          for (unsigned int i = begin; i < end; ++i) {
            auto &row = rows[i];
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
            const auto diagonal = std::lower_bound(row.begin(), row.end(), i);
            if (diagonal != row.end() && *diagonal == i)
              std::rotate(row.begin(), diagonal, diagonal + 1);
            else
              row.insert(row.begin(), i);
          }
        },
        4096);

    std::vector<std::size_t> offsets(n_locally_relevant_dofs + 1, 0);
    for (unsigned int i = 0; i < n_locally_relevant_dofs; ++i)
      offsets[i + 1] = offsets[i] + rows[i].size();
    const std::size_t n_nonzero_elements = offsets.back();

    /* Position of the entry (i, j), or invalid_position: */
    constexpr auto invalid_position = std::numeric_limits<std::size_t>::max();
    const auto position_of = [&](const unsigned int i, const unsigned int j) {
      if (i == j)
        return offsets[i];
      const auto first = rows[i].begin() + 1;
      const auto it = std::lower_bound(first, rows[i].end(), j);
      if (it == rows[i].end() || *it != j)
        return invalid_position;
      return offsets[i] + (it - rows[i].begin());
    };

    Assert(n_internal_dofs <= n_locally_relevant_dofs,
           dealii::ExcInternalError());
    Assert(n_internal_dofs % warp_size == 0, dealii::ExcInternalError());
    Assert(n_internal_dofs <= n_locally_owned_dofs_,
           dealii::ExcInternalError());
    Assert(n_locally_owned_dofs_ <= n_locally_relevant_dofs,
           dealii::ExcInternalError());

    AssertThrow(
        n_nonzero_elements < std::numeric_limits<unsigned int>::max(),
        dealii::ExcMessage(
            "Transposed indices only support up to 4 billion matrix entries "
            "per MPI rank. Try to split into smaller problems with MPI"));

    /* Allocate memory: */

    row_starts_.reinit(n_locally_relevant_dofs + 1);
    column_indices_.reinit(n_nonzero_elements);
    indices_transposed_.reinit(n_nonzero_elements);
    allocate_storage<dealii::MemorySpace::Host>();

    auto *row_starts = row_starts_.data();
    auto *column_indices = column_indices_.data();
    auto *indices_transposed = indices_transposed_.data();

    /*
     * Position of the entry (row, column) in the transposed row. Rows of
     * the vectorized part are stored interleaved in warps of warp_size rows:
     */
    const auto transposed_position = [&](const unsigned int row,
                                         const unsigned int column) {
      const std::size_t position = position_of(column, row);
      if (column >= n_internal_dofs)
        return position;
      const unsigned int my_row_length = rows[column].size();
      const std::size_t position_diag = offsets[column];
      const std::size_t pos_within_row = position - position_diag;
      const unsigned int lane = column % warp_size;
      return position - lane * my_row_length - pos_within_row + lane +
             pos_within_row * warp_size;
    };

    /*
     * Compute all row starts first. This allows us to fill the rows in
     * parallel. All rows of a warp in the vectorized part have the same
     * length:
     */

    row_starts[0] = 0;
    for (unsigned int i = 0; i < n_internal_dofs; i += warp_size)
      row_starts[i / warp_size + 1] =
          row_starts[i / warp_size] + warp_size * rows[i].size();

    row_starts[n_internal_dofs] = row_starts[n_internal_dofs / warp_size];

    for (unsigned int i = n_internal_dofs; i < n_locally_relevant_dofs; ++i)
      row_starts[i + 1] = row_starts[i] + rows[i].size();

    Assert(row_starts[n_locally_relevant_dofs] == column_indices_.size(),
           dealii::ExcInternalError());

    /* Vectorized part: */

    dealii::parallel::apply_to_subranges(
        0u,
        n_internal_dofs / warp_size,
        [&](const unsigned int begin, const unsigned int end) {
          for (unsigned int warp = begin; warp < end; ++warp) {
            const unsigned int i = warp * warp_size;
            auto position = row_starts[warp];

            for (unsigned int m = 0; m < rows[i].size(); ++m)
              for (unsigned int k = 0; k < warp_size; ++k, ++position) {
                const unsigned int column = rows[i + k][m];
                column_indices[position] = column;
                indices_transposed[position] =
                    transposed_position(i + k, column);
              }
          }
        },
        64);

    /* Rest: */

    dealii::parallel::apply_to_subranges(
        n_internal_dofs,
        static_cast<unsigned int>(n_locally_relevant_dofs),
        [&](const unsigned int begin, const unsigned int end) {
          for (unsigned int i = begin; i < end; ++i) {
            auto position = row_starts[i];
            for (const auto column : rows[i]) {
              column_indices[position] = column;
              indices_transposed[position] = transposed_position(i, column);
              ++position;
            }
          }
        },
        4096);


    /*
     * Compute the data exchange pattern:
     */

    receive_targets_.clear();
    send_targets_.clear();

    std::vector<ExchangeDescription> entries_to_be_sent;

    if (n_locally_relevant_dofs > n_locally_owned_dofs_) {
      const unsigned int mpi_tag =
          dealii::Utilities::MPI::internal::Tags::partitioner_export_start + 0;

      const auto &ghost_targets = partitioner->ghost_targets();
      const auto &import_targets = partitioner->import_targets();
      const auto &mpi_communicator = partitioner->get_mpi_communicator();

      const unsigned int n_requests =
          ghost_targets.size() + import_targets.size();
      std::vector<MPI_Request> requests(n_requests);

      /*
       * Set up receive targets.
       *
       * We receive our local ghost rows from MPI ranks in the ghost range
       * of the (scalar) partitioner. We receive our entire local ghost row
       * from the owning MPI rank. We have to navigate one detail, though.
       * Our local view of the ghost row is a subset of the full row of the
       * owning rank. We thus have to communicate to the owning rank how
       * many entries and what indices we are expecting.
       *
       * First, set up the receive_targets_ vector and send the cummulative
       * row size to the owning MPI rank:
       */

      receive_targets_.resize(ghost_targets.size());
      for (unsigned int p = 0; p < receive_targets_.size(); ++p) {
        receive_targets_[p].first = ghost_targets[p].first;
      }

      {
        /* Index into ghost targets: */
        unsigned int ghost_targets_index = 0;
        /* Current and previous index into ghost range of sparsity pattern: */
        unsigned int index = 0;
        unsigned int previous_index = 0;

        unsigned int row_count = 0;
        for (unsigned int i = n_locally_owned_dofs_;
             i < n_locally_relevant_dofs;
             ++i) {
          index += rows[i].size();
          ++row_count;
          const auto ghost_target = ghost_targets[ghost_targets_index];
          if (row_count == ghost_target.second) {
            receive_targets_[ghost_targets_index].second = index;

            unsigned int n_entries = index - previous_index;
            const int ierr = MPI_Isend(
                &n_entries,
                1,
                dealii::Utilities::MPI::mpi_type_id_for_type<unsigned int>,
                ghost_target.first,
                mpi_tag,
                mpi_communicator,
                &requests[ghost_targets_index]);
            AssertThrowMPI(ierr);

            /* Update indices: */
            ++ghost_targets_index;
            previous_index = index;
            row_count = 0;
          }
        }

        Assert(ghost_targets_index == partitioner->ghost_targets().size(),
               dealii::ExcInternalError());
      }


      /*
       * Set up send targets.
       *
       * First receive the number of entries that we will need to send.
       */

      std::vector<unsigned int> send_ranges(import_targets.size());
      for (unsigned int p = 0; p < import_targets.size(); ++p) {
        const int ierr = MPI_Irecv(
            &send_ranges[p],
            1,
            dealii::Utilities::MPI::mpi_type_id_for_type<unsigned int>,
            import_targets[p].first,
            mpi_tag,
            mpi_communicator,
            &requests[ghost_targets.size() + p]);
        AssertThrowMPI(ierr);
      }

      {
        const int ierr =
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        AssertThrowMPI(ierr);
      }

      /*
       * Now, that the owning rank knows the number of entries we request
       * we can send the actual index pairs (i_global, j_global) that we
       * require.
       */

      std::vector<dealii::types::global_dof_index> requested_entries;
      {
        /* Index into ghost targets: */
        unsigned int ghost_targets_index = 0;

        unsigned int row_count = 0;
        for (unsigned int i = n_locally_owned_dofs_;
             i < n_locally_relevant_dofs;
             ++i) {
          const auto i_global = partitioner_->local_to_global(i);
          for (const auto j : rows[i]) {
            const auto j_global = partitioner_->local_to_global(j);
            requested_entries.push_back(i_global);
            requested_entries.push_back(j_global);
          }

          ++row_count;
          if (row_count == ghost_targets[ghost_targets_index].second) {
            /* Update indices: */
            ++ghost_targets_index;
            row_count = 0;
          }
        }

        Assert(ghost_targets_index == partitioner->ghost_targets().size(),
               dealii::ExcInternalError());

#ifdef DEBUG
        const auto ghost_offset = row_starts[n_locally_owned_dofs_];
        const auto n_nonzero_elements = row_starts[n_locally_relevant_dofs];
        Assert(requested_entries.size() ==
                   2 * (n_nonzero_elements - ghost_offset),
               dealii::ExcInternalError());
#endif

        for (unsigned int p = 0; p < receive_targets_.size(); ++p) {
          const auto request_offset =
              p == 0 ? 0 : receive_targets_[p - 1].second;
          const auto request_size = receive_targets_[p].second - request_offset;

          const int ierr =
              MPI_Isend(requested_entries.data() + 2 * request_offset,
                        2 * request_size,
                        dealii::Utilities::MPI::mpi_type_id_for_type<
                            dealii::types::global_dof_index>,
                        receive_targets_[p].first,
                        mpi_tag,
                        mpi_communicator,
                        &requests[p]);
          AssertThrowMPI(ierr);
        }
      }

      /*
       * Accumulate all requests we received from other ranks:
       */

      send_targets_.resize(import_targets.size());

      const unsigned int n_entries_to_be_sent =
          std::accumulate(send_ranges.begin(), send_ranges.end(), 0);
      std::vector<dealii::types::global_dof_index> entries_buffer(
          2 * n_entries_to_be_sent /*!*/);

      {
        /* Index into entries_to_be_sent: */
        unsigned int index = 0;

        for (unsigned int p = 0; p < send_targets_.size(); ++p) {
          const auto n_entries = send_ranges[p];

          const int ierr =
              MPI_Irecv(entries_buffer.data() + 2 * index /*!*/,
                        2 * n_entries /*!*/,
                        dealii::Utilities::MPI::mpi_type_id_for_type<
                            dealii::types::global_dof_index>,
                        import_targets[p].first,
                        mpi_tag,
                        mpi_communicator,
                        &requests[ghost_targets.size() + p]);
          AssertThrowMPI(ierr);

          index += n_entries;
          send_targets_[p].first = import_targets[p].first;
          send_targets_[p].second = index;
        }
      }

      {
        const int ierr =
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        AssertThrowMPI(ierr);
      }

      entries_to_be_sent.reserve(n_entries_to_be_sent);
      for (unsigned int e = 0; e < n_entries_to_be_sent; ++e) {
        const auto i_global = entries_buffer[2 * e];
        const auto j_global = entries_buffer[2 * e + 1];
        const auto i = partitioner_->global_to_local(i_global);
        const auto j = partitioner_->global_to_local(j_global);

        const std::size_t position = position_of(i, j);
        Assert(
            position != invalid_position,
            dealii::ExcMessage("Inconsistent global view of sparsity pattern: "
                               "the requested column index is not present on "
                               "the row stored on the owning MPI rank."));

        const std::size_t position_diag = offsets[i];
        const std::size_t position_within_row = position - position_diag;

        entries_to_be_sent.emplace_back(ExchangeDescription{
            i, static_cast<unsigned int>(position_within_row)});
      }
    }

    /*
     * FIXME: entries_to_be_sent_ should inherit the same "transfer_policy"
     * at this point. Unfortunately, we do not support calling
     * copy_to_memory_space<>(), or move_to_memory_space<>() of dependent
     * data fields in an elegant way... Work around this issue by setting
     * the transfer policy to "implicit_transfers".
     */
    entries_to_be_sent_.reinit(entries_to_be_sent.size(),
                               TransferPolicy::implicit_transfers);
    std::copy(entries_to_be_sent.begin(),
              entries_to_be_sent.end(),
              entries_to_be_sent_.view());

    this->reset_residency(/*host*/ true,
                          /*default*/ !have_separate_memory_spaces);

    /* The transfer policy is selected last, see MirroredResidency: */
    this->set_transfer_policy(transfer_policy);
  }
} // namespace ryujin
