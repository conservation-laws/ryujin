//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>
#include <convenience_macros.h>

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/config.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

namespace ryujin
{
  template <int simd_length,
            typename MemorySpace = dealii::MemorySpace::Host::kokkos_space>
  class SparsityPatternView;


  /**
   * A specialized sparsity pattern for efficient, vectorized SIMD access.
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
  class SparsityPattern : public SparsityPatternView<simd_length>
  {
  public:
    /**
     * Constructor, initialization, access.
     */
    //@{

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
            &partitioner,
        bool symmetrize_ghost_range = true);

    /**
     * Reinit function that reinitializes the SIMD sparsity pattern for a
     * given sparsity pattern template, an MPI partitioner and the number
     * of (regular) internal dofs.
     *
     * @note If the @p symmetrize_ghost_range parameter is set to true,
     * then all transpose entries (j, i) are added to the sparsity pattern,
     * where i is within the locally owned range. This access is required
     * for our stencil based methods.
     */
    void reinit(const unsigned int n_internal_dofs,
                const dealii::DynamicSparsityPattern &sparsity,
                const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
                    &partitioner,
                bool symmetrize_ghost_range = true);

    /**
     * Return a (read only) view on the sparsity pattern for the selected
     * memory space.
     */
    template <typename MemorySpace>
    SparsityPatternView<simd_length, MemorySpace> get_view() const;

    ACCESSOR_READ_ONLY_NO_DEREFERENCE(partitioner);

    /**
     * Array listing all (locally owned) entries as a pair {row,
     * position_within_column}, potentially duplicated, and arranged
     * consecutively by send targets.
     */
    ACCESSOR_READ_ONLY(entries_to_be_sent);

    /**
     * All send targets stored as a pair consisting of an MPI rank (first
     * entry) and a corresponding index range into entries_to_be_sent given
     * by the half open range [send_targets[p-1].second, send_targets[p])
     */
    ACCESSOR_READ_ONLY(send_targets);

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
    ACCESSOR_READ_ONLY(receive_targets);

  private:
    //@}
    /**
     * @name Internal fields, methods, and friends
     */
    //@{

    std::shared_ptr<const dealii::Utilities::MPI::Partitioner> partitioner_;

    unsigned int n_internal_dofs_;
    unsigned int n_locally_owned_dofs_;

    using HostSpace = dealii::MemorySpace::Host::kokkos_space;
    Kokkos::View<unsigned int *, HostSpace> row_starts_host_;
    Kokkos::View<unsigned int *, HostSpace> column_indices_host_;
    Kokkos::View<unsigned int *, HostSpace> indices_transposed_host_;

    using DefaultSpace = dealii::MemorySpace::Default::kokkos_space;
    Kokkos::View<unsigned int *, DefaultSpace> row_starts_default_;
    Kokkos::View<unsigned int *, DefaultSpace> column_indices_default_;
    Kokkos::View<unsigned int *, DefaultSpace> indices_transposed_default_;

    std::vector<std::pair<unsigned int, unsigned int>> entries_to_be_sent_;
    std::vector<std::pair<unsigned int, unsigned int>> send_targets_;
    std::vector<std::pair<unsigned int, unsigned int>> receive_targets_;

    template <int, typename>
    friend class SparsityPatternView;
    //@}
  };


  /**
   * This class models a "view" of the sparsity pattern that lives in the
   * host or device memory space. It provides a number of methods for
   * iterating over the sparsity pattern and offset computation.
   *
   * @note This class is designed to be captured by value in computation
   * loops with access to either the host or device memory space. As such
   * we do not store a reference to the underlying SparsityPattern but
   * rather raw pointers into the corresponding memory. The view is only
   * valid as long as the underlying SparsityPattern object is not
   * modified.
   */
  template <int simd_length, typename MemorySpace>
  class SparsityPatternView
  {
  public:
    SparsityPatternView() = default;

    SparsityPatternView(const SparsityPattern<simd_length> &sparsity_pattern);

    void reinit(const SparsityPattern<simd_length> &sparsity_pattern);

    DEAL_II_HOST_DEVICE
    unsigned int n_internal_dofs() const;

    DEAL_II_HOST_DEVICE
    unsigned int n_locally_owned_dofs() const;

    /**
     * Return the "stride size" of a given row index. The function returns
     * simd_length for all indices in the range [0, n_internal_dofs) and 1
     * otherwise.
     */
    DEAL_II_HOST_DEVICE
    unsigned int stride_of_row(const unsigned int row) const;

    /**
     * Return a pointer to the array of column indices for the given row,
     * i.e., for a given row index i:
     * ```
     *   const unsigned int *js = sparsity_simd.columns(i);
     * ```
     * is a pointer to the column index j (or column indices *js when
     * SIMD vectorized).
     */
    DEAL_II_HOST_DEVICE
    const unsigned int *columns(const unsigned int row) const;

    /**
     * Return the row length of a given row index.
     */
    DEAL_II_HOST_DEVICE
    unsigned int row_length(const unsigned int row) const;

    /**
     * The total number of rows of the given sparsity pattern.
     */
    DEAL_II_HOST_DEVICE
    unsigned int n_rows() const;

    /**
     * The total number of nonzero elements of the given sparsity pattern.
     */
    DEAL_II_HOST_DEVICE
    unsigned int n_nonzero_elements() const;

    /**
     * Given a row index, an index for the column (within [0,
     * row_length(row)), and a component index return the position of the
     * matrix entry in the data array.
     */
    template <unsigned int n_components = 1>
    DEAL_II_HOST_DEVICE unsigned int
    offset(const unsigned int row,
           const unsigned int position_within_column,
           const unsigned int component = 0) const;

    /**
     * Specialized version of the function above that computes the offset
     * only for the internal part and pointing to component 0. This variant
     * avoids a number of index computations and an if statement.
     *
     * @pre row must be within the internal index range.
     */
    template <unsigned int n_components = 1>
    DEAL_II_HOST_DEVICE unsigned int
    offset_internal(const unsigned int row,
                    const unsigned int position_within_column) const;

    /**
     * Given a row index, an index for the column (within [0,
     * row_length(row)), and a component index return the position of the
     * *transposed* matrix entry in the data array.
     */
    template <unsigned int n_components = 1>
    DEAL_II_HOST_DEVICE unsigned int
    transposed_offset(const unsigned int row,
                      const unsigned int position_within_column,
                      const unsigned int component = 0) const;

    /**
     * Specialized version of the function above that computes the offset
     * only for the internal part and pointing to component 0. This variant
     * avoids a number of index computations and an if statement.
     *
     * @pre row must be within the internal index range.
     */
    template <unsigned int n_components = 1>
    DEAL_II_HOST_DEVICE const unsigned int *
    transposed_offset_internal(const unsigned int row,
                               const unsigned int position_within_column) const;

    /**
     * Return an offset pointing to the first element of the ghost range,
     * i.e., the value corresponding to
     * `offset(sparsity_->n_locally_owned_dofs(), 0, 0)`.
     *
     * @note If the sparsity pattern does not contain a ghost range, then
     * the offset points one element past the data array of the sparse
     * matrix.
     */
    template <unsigned int n_components = 1>
    DEAL_II_HOST_DEVICE unsigned int ghost_offset() const;

  private:
    unsigned int n_internal_dofs_;
    unsigned int n_locally_owned_dofs_;

    Kokkos::View<const unsigned int *, MemorySpace> row_starts_;
    Kokkos::View<const unsigned int *, MemorySpace> column_indices_;
    Kokkos::View<const unsigned int *, MemorySpace> indices_transposed_;
  };


#ifndef DOXYGEN
  /*
   * -------------------------------------------------------------------------
   * Inline function definitions
   * -------------------------------------------------------------------------
   */


  template <int simd_length>
  template <typename MemorySpace>
  SparsityPatternView<simd_length, MemorySpace>
  SparsityPattern<simd_length>::get_view() const
  {
    return SparsityPatternView<simd_length, MemorySpace>(*this);
  }


  template <int simd_length, typename MemorySpace>
  SparsityPatternView<simd_length, MemorySpace>::SparsityPatternView(
      const SparsityPattern<simd_length> &sparsity_pattern)
  {
    reinit(sparsity_pattern);
  }


  template <int simd_length, typename MemorySpace>
  void SparsityPatternView<simd_length, MemorySpace>::reinit(
      const SparsityPattern<simd_length> &sparsity_pattern)
  {
    n_internal_dofs_ = sparsity_pattern.n_internal_dofs_;
    n_locally_owned_dofs_ = sparsity_pattern.n_locally_owned_dofs_;

    using HostSpace = dealii::MemorySpace::Host::kokkos_space;
    using DefaultSpace = dealii::MemorySpace::Default::kokkos_space;

    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected Kokkos memory space");

    if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
      row_starts_ = sparsity_pattern.row_starts_host_;
      column_indices_ = sparsity_pattern.column_indices_host_;
      indices_transposed_ = sparsity_pattern.indices_transposed_host_;
    } else {
      row_starts_ = sparsity_pattern.row_starts_default_;
      column_indices_ = sparsity_pattern.column_indices_default_;
      indices_transposed_ = sparsity_pattern.indices_transposed_default_;
    }
  }


  template <int simd_length, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<simd_length, MemorySpace>::n_internal_dofs() const
  {
    return n_internal_dofs_;
  }


  template <int simd_length, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<simd_length, MemorySpace>::n_locally_owned_dofs() const
  {
    return n_locally_owned_dofs_;
  }


  template <int simd_length, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<simd_length, MemorySpace>::stride_of_row(
      const unsigned int row) const
  {
    AssertIndexRange(row, n_rows());

    if (row < n_internal_dofs_)
      return simd_length;
    else
      return 1;
  }


  template <int simd_length, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE const unsigned int *
  SparsityPatternView<simd_length, MemorySpace>::columns(
      const unsigned int row) const
  {
    AssertIndexRange(row, n_rows());

    if (row < n_internal_dofs_)
      return column_indices_.data() + row_starts_(row / simd_length) +
             row % simd_length;
    else
      return column_indices_.data() + row_starts_(row);
  }


  template <int simd_length, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<simd_length, MemorySpace>::row_length(
      const unsigned int row) const
  {
    AssertIndexRange(row, n_rows());

    if (row < n_internal_dofs_) {
      const unsigned int simd_row = row / simd_length;
      return (row_starts_(simd_row + 1) - row_starts_(simd_row)) / simd_length;
    } else {
      return row_starts_(row + 1) - row_starts_(row);
    }
  }


  template <int simd_length, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<simd_length, MemorySpace>::n_rows() const
  {
    Assert(row_starts_.size() > 0, dealii::ExcNotInitialized());

    return row_starts_.size() - 1;
  }


  template <int simd_length, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<simd_length, MemorySpace>::n_nonzero_elements() const
  {
    Assert(row_starts_.size() > 0, dealii::ExcNotInitialized());

    return row_starts_(row_starts_.size() - 1);
  }


  template <int simd_length, typename MemorySpace>
  template <unsigned int n_components>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<simd_length, MemorySpace>::offset(
      const unsigned int row,
      const unsigned int position_within_column,
      const unsigned int comp) const
  {
    AssertIndexRange(row, n_rows());
    AssertIndexRange(position_within_column, row_length(row));
    AssertIndexRange(comp, n_components);

    const unsigned int simd_row = row / simd_length;
    const unsigned int simd_offset = row % simd_length;

    if (row < n_internal_dofs_) {
      const unsigned int scalar_offset =
          row_starts_(simd_row) + position_within_column * simd_length;
      return scalar_offset * n_components + comp * simd_length + simd_offset;

    } else {
      const unsigned int scalar_offset =
          row_starts_(row) + position_within_column;

      return scalar_offset * n_components + comp;
    }
  }


  template <int simd_length, typename MemorySpace>
  template <unsigned int n_components>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<simd_length, MemorySpace>::transposed_offset(
      const unsigned int row,
      const unsigned int position_within_column,
      const unsigned int component) const
  {
    AssertIndexRange(row, n_rows());
    AssertIndexRange(position_within_column, row_length(row));
    AssertIndexRange(component, n_components);

    // Compute the transposed index from the (scalar) numbering stored in
    // the sparsity pattern...
    const unsigned int scalar_offset = offset(row, position_within_column);
    const unsigned int transposed_scalar_offset =
        indices_transposed_(scalar_offset);

    // ... and reconstruct the proper index for a view with n_components:
    const unsigned int column_index = column_indices_(scalar_offset);

    unsigned int transposed_offset = transposed_scalar_offset;
    if constexpr (n_components > 1) {
      if (column_index < n_internal_dofs_) {
        transposed_offset = //
            transposed_offset / simd_length * simd_length * n_components +
            transposed_offset % simd_length;
        return transposed_offset + component * simd_length;

      } else {

        transposed_offset *= n_components;
        return transposed_offset + component;
      }

    } else {

      return transposed_offset;
    }
  }


  template <int simd_length, typename MemorySpace>
  template <unsigned int n_components>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<simd_length, MemorySpace>::offset_internal(
      const unsigned int row, const unsigned int position_within_column) const
  {
    AssertIndexRange(row, n_rows());
    AssertIndexRange(position_within_column, row_length(row));
    AssertIndexRange(row, n_internal_dofs_);

    const unsigned int simd_row = row / simd_length;

    Assert(row % simd_length == 0,
           dealii::ExcMessage(
               "Access only supported for rows at the SIMD granularity"));

    const unsigned int scalar_offset =
        row_starts_(simd_row) + position_within_column * simd_length;

    return scalar_offset * n_components;
  }


  template <int simd_length, typename MemorySpace>
  template <unsigned int n_components>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE const unsigned int *
  SparsityPatternView<simd_length, MemorySpace>::transposed_offset_internal(
      const unsigned int row, const unsigned int position_within_column) const
  {
    static_assert(n_components == 1,
                  "Vectorized transposed access to multiple components is not "
                  "yet implemented.");
    AssertIndexRange(row, row_starts_.size() - 1);
    AssertIndexRange(position_within_column, row_length(row));
    AssertIndexRange(row, n_internal_dofs_);

    const unsigned int simd_row = row / simd_length;

    Assert(row % simd_length == 0,
           dealii::ExcMessage(
               "Access only supported for rows at the SIMD granularity"));

    const unsigned int scalar_offset =
        row_starts_(simd_row) + position_within_column * simd_length;

    // n_components == 1
    return indices_transposed_.data() + scalar_offset;
  }


  template <int simd_length, typename MemorySpace>
  template <unsigned int n_components>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<simd_length, MemorySpace>::ghost_offset() const
  {
    const auto scalar_offset = row_starts_(n_locally_owned_dofs_);
    return scalar_offset * n_components;
  }


#endif
} // namespace ryujin
