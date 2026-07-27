//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include "simd.h"
#include "sparsity_pattern.h"

#include <deal.II/base/vectorization.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>

namespace ryujin
{
  template <int simd_length>
  SparsityPattern<simd_length>::SparsityPattern()
      : n_internal_dofs_(0)
  {
  }


  template <int simd_length>
  SparsityPattern<simd_length>::SparsityPattern(
      const unsigned int n_internal_dofs,
      const dealii::DynamicSparsityPattern &sparsity,
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          &partitioner,
      bool symmetrize_ghost_range)
  {
    reinit(n_internal_dofs, sparsity, partitioner, symmetrize_ghost_range);
  }


  template <int simd_length>
  void SparsityPattern<simd_length>::reinit(
      const unsigned int n_internal_dofs,
      const dealii::DynamicSparsityPattern &dsp,
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          &partitioner,
      const bool symmetrize_ghost_range)
  {
    this->n_internal_dofs_ = n_internal_dofs;
    this->n_locally_owned_dofs_ = partitioner->locally_owned_size();
    this->partitioner_ = partitioner;

    const auto n_locally_relevant_dofs =
        partitioner->locally_owned_size() + partitioner->n_ghost_indices();

    /*
     * First, create a static sparsity pattern in local indexing.
     */

    dealii::DynamicSparsityPattern dsp_local(n_locally_relevant_dofs,
                                             n_locally_relevant_dofs);
    for (unsigned int i = 0; i < n_locally_relevant_dofs; ++i) {
      const auto global_row = partitioner->local_to_global(i);
      for (auto it = dsp.begin(global_row); it != dsp.end(global_row); ++it) {
        const auto global_column = it->column();
        const auto j = partitioner->global_to_local(global_column);
        dsp_local.add(i, j);

        if (symmetrize_ghost_range && //
            i < n_locally_owned_dofs_ && j >= n_locally_owned_dofs_)
          dsp_local.add(j, i);
      }
    }

    dealii::SparsityPattern sparsity;
    sparsity.copy_from(dsp_local);

    Assert(n_internal_dofs <= sparsity.n_rows(), dealii::ExcInternalError());
    Assert(n_internal_dofs % simd_length == 0, dealii::ExcInternalError());
    Assert(n_internal_dofs <= n_locally_owned_dofs_,
           dealii::ExcInternalError());
    Assert(n_locally_owned_dofs_ <= sparsity.n_rows(),
           dealii::ExcInternalError());

    AssertThrow(
        sparsity.n_nonzero_elements() <
            std::numeric_limits<unsigned int>::max(),
        dealii::ExcMessage(
            "Transposed indices only support up to 4 billion matrix entries "
            "per MPI rank. Try to split into smaller problems with MPI"));

    /* Allocate memory: */

    using KokkosHost = dealii::MemorySpace::Host::kokkos_space;
    using KokkosDefault = dealii::MemorySpace::Default::kokkos_space;
    using Aligned = Kokkos::MemoryTraits<Kokkos::Aligned>;

    row_starts_host_ = Kokkos::View<unsigned int *, KokkosHost, Aligned>(
        "sparsity_pattern_row_starts", sparsity.n_rows() + 1);

    column_indices_host_ = Kokkos::View<unsigned int *, KokkosHost, Aligned>(
        "sparsity_pattern_column_indices", sparsity.n_nonzero_elements());

    indices_transposed_host_ =
        Kokkos::View<unsigned int *, KokkosHost, Aligned>(
            "sparsity_pattern_column_indices", sparsity.n_nonzero_elements());

    /* Vectorized part: */

    row_starts_host_[0] = 0;

    unsigned int *col_ptr = column_indices_host_.data();
    unsigned int *transposed_ptr = indices_transposed_host_.data();

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

      row_starts_host_[i / simd_length + 1] =
          col_ptr - column_indices_host_.data();
    }

    /* Rest: */

    row_starts_host_[n_internal_dofs] =
        row_starts_host_[n_internal_dofs / simd_length];

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
      row_starts_host_[i + 1] = col_ptr - column_indices_host_.data();
    }

#ifdef DEBUG
    const auto distance = std::distance(column_indices_host_.data(), col_ptr);
    Assert(static_cast<std::size_t>(distance) == column_indices_host_.size(),
           dealii::ExcInternalError());
#endif

    /*
     * Compute the data exchange pattern:
     */

    receive_targets_.clear();
    send_targets_.clear();
    entries_to_be_sent_.clear();

    if (sparsity.n_rows() > n_locally_owned_dofs_) {
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
        for (unsigned int i = n_locally_owned_dofs_; i < sparsity.n_rows();
             ++i) {
          index += sparsity.row_length(i);
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
        for (unsigned int i = n_locally_owned_dofs_; i < sparsity.n_rows();
             ++i) {
          const auto i_global = partitioner_->local_to_global(i);
          for (auto idx = sparsity.begin(i); idx != sparsity.end(i); ++idx) {
            const unsigned int j = idx->column();
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
        const auto ghost_offset = row_starts_host_(n_locally_owned_dofs_);
        const auto n_nonzero_elements =
            row_starts_host_(row_starts_host_.size() - 1);
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

      entries_to_be_sent_.clear();
      for (unsigned int e = 0; e < n_entries_to_be_sent; ++e) {
        const auto i_global = entries_buffer[2 * e];
        const auto j_global = entries_buffer[2 * e + 1];
        const auto i = partitioner_->global_to_local(i_global);
        const auto j = partitioner_->global_to_local(j_global);

        const std::size_t position = sparsity(i, j);
        Assert(
            position != sparsity.invalid_entry,
            dealii::ExcMessage("Inconsistent global view of sparsity pattern: "
                               "the requested column index is not present on "
                               "the row stored on the owning MPI rank."));

        const std::size_t position_diag = sparsity(i, i);
        const std::size_t position_within_row = position - position_diag;

        entries_to_be_sent_.emplace_back(i, position_within_row);
      }
    }

    /*
     * Copy data over to device and initialize the default host view:
     */

    row_starts_default_ = Kokkos::create_mirror_view_and_copy(
        typename KokkosDefault::execution_space(), row_starts_host_);

    column_indices_default_ = Kokkos::create_mirror_view_and_copy(
        typename KokkosDefault::execution_space(), column_indices_host_);

    indices_transposed_default_ = Kokkos::create_mirror_view_and_copy(
        typename KokkosDefault::execution_space(), indices_transposed_host_);

    SparsityPatternView<simd_length>::reinit(*this);
  }
} // namespace ryujin
