//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

namespace ryujin
{
  template <typename Number,
            int n_components = 1,
            int simd_length = dealii::VectorizedArray<Number>::size()>
  class SparseMatrix;

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
  class SparsityPattern
  {
  public:
    /**
     * Default constructor.
     */
    SparsityPattern();

    /**
     * Constructor taking a sparsity pattern template, an MPI partitioner
     * and the number of (regular) internal dofs as an argument. The
     * constructor calls SparsityPattern::reinit() internally.
     */
    SparsityPattern(
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

  protected:
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
     * All receive targets are stored as a pair consisting of an MPI rank
     * (first entry) and a corresponding index range into the (serial)
     * data array given by the half open range
     * [receive_targets[p-1].second, receive_targets[p].second).
     *
     * Note, that indices into the data array start with the "locally
     * relevant", or "ghost range" offset by n_locally_owned_dofs and
     * multiplied by the number of components stored by the (vector valued)
     * matrix.
     */
    std::vector<std::pair<unsigned int, unsigned int>> receive_targets;

    MPI_Comm mpi_communicator;

    template <typename, int, int>
    friend class SparseMatrix;
  };


#ifndef DOXYGEN
    /*
     * -------------------------------------------------------------------------
     * Inline function definitions
     * -------------------------------------------------------------------------
     */


  template <int simd_length>
  DEAL_II_ALWAYS_INLINE inline unsigned int
  SparsityPattern<simd_length>::stride_of_row(const unsigned int row) const
  {
    AssertIndexRange(row, row_starts.size());

    if (row < n_internal_dofs)
      return simd_length;
    else
      return 1;
  }


  template <int simd_length>
  DEAL_II_ALWAYS_INLINE inline const unsigned int *
  SparsityPattern<simd_length>::columns(const unsigned int row) const
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
  SparsityPattern<simd_length>::row_length(const unsigned int row) const
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
  SparsityPattern<simd_length>::n_rows() const
  {
    return row_starts.size() - 1;
  }


  template <int simd_length>
  DEAL_II_ALWAYS_INLINE inline std::size_t
  SparsityPattern<simd_length>::n_nonzero_elements() const
  {
    Assert(row_starts.size() > 0, dealii::ExcNotInitialized());

    return row_starts.back();
  }

#endif
} // namespace ryujin
