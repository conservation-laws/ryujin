//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>
#include <convenience_macros.h>

#include "gpu.h"

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/config.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

namespace ryujin
{
  template <int warp_size, typename MemorySpace = dealii::MemorySpace::Host>
  class SparsityPatternView;


  /**
   * A specialized sparsity pattern for efficient, vectorized SIMD and GPU
   * access.
   *
   * In the vectorized row index region [0, n_internal_dofs) we store data
   * as an array-of-struct-of-array type as follows:
   *
   *  - At the innermost 'array' level, we group data from warp_size rows
   *    (a "warp") contiguously in memory, using a given column block as
   *    determined by the sparsity pattern.
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
   *
   * @ingroup LinearAlgebra
   */
  template <int warp_size>
  class SparsityPattern : public MirroredStorage<SparsityPattern<warp_size>>
  {
  public:
    /**
     * @name Types
     */
    //@{

    /**
     * A structure describing a single matrix entry (given by a row index
     * and an index for the column within that row) that has to be sent to
     * a neighboring MPI rank during an update_ghost_rows(), or compress()
     * operation.
     */
    struct ExchangeDescription {
      unsigned int row;
      unsigned int column_index;
    };

    //@}
    /**
     * @name Constructor and initialization
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
        bool symmetrize_ghost_range = true,
        const TransferPolicy transfer_policy =
            TransferPolicy::explicit_transfers);

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
                bool symmetrize_ghost_range = true,
                const TransferPolicy transfer_policy =
                    TransferPolicy::explicit_transfers);

    /**
     * Return a (read only) view on the sparsity pattern for the selected
     * memory space.
     */
    template <typename MemorySpace = dealii::MemorySpace::Host>
    SparsityPatternView<warp_size, MemorySpace> view() const;

    ACCESSOR_READ_ONLY_NO_DEREFERENCE(partitioner);

    /**
     * An array of ExchangeDescription structures listing all (locally
     * owned) entries, potentially duplicated, and arranged consecutively
     * by send targets. The function returns a reference to a Mirrored
     * object holding {row, column_index} pairs that can be accessed on
     * both memory spaces.
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

    using KokkosHost = dealii::MemorySpace::Host::kokkos_space;
    mutable Kokkos::View<unsigned int *, KokkosHost> row_starts_host_;
    mutable Kokkos::View<unsigned int *, KokkosHost> column_indices_host_;
    mutable Kokkos::View<unsigned int *, KokkosHost> indices_transposed_host_;

    using KokkosDefault = dealii::MemorySpace::Default::kokkos_space;
    mutable Kokkos::View<unsigned int *, KokkosDefault> row_starts_default_;
    mutable Kokkos::View<unsigned int *, KokkosDefault> column_indices_default_;
    mutable Kokkos::View<unsigned int *, KokkosDefault>
        indices_transposed_default_;

    Mirrored<ExchangeDescription *> entries_to_be_sent_{
        "sparsity_pattern_entries_to_be_sent"};

    std::vector<std::pair<unsigned int, unsigned int>> send_targets_;
    std::vector<std::pair<unsigned int, unsigned int>> receive_targets_;

    /*
     * Storage primitives used by the MirroredStorage base class:
     */

    template <typename MemorySpace>
    void allocate_storage() const;

    template <typename To, typename From>
    void deep_copy_storage() const;

    template <typename MemorySpace>
    void deallocate_storage();


    friend class MirroredStorage<SparsityPattern<warp_size>>;

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
   *
   * @ingroup LinearAlgebra
   */
  template <int warp_size, typename MemorySpace>
  class SparsityPatternView
  {
  public:
    /**
     * @name Constructor and initialization
     */
    //@{

    SparsityPatternView() = default;

    SparsityPatternView(const SparsityPattern<warp_size> &sparsity_pattern);

    void reinit(const SparsityPattern<warp_size> &sparsity_pattern);

    //@}
    /**
     * @name General information about the sparsity pattern
     */
    //@{

    /**
     * The number of locally internal rows of the sparsity pattern that are
     * stored in optimized "array-of-struct-of-array" format.
     */
    DEAL_II_HOST_DEVICE
    unsigned int n_internal_dofs() const;

    /**
     * The number of locally owned rows of the sparsity pattern.
     */
    DEAL_II_HOST_DEVICE
    unsigned int n_locally_owned_dofs() const;

    /**
     * The total number of rows of the given sparsity pattern. This number
     * comprises all locally owned rows and the ghost row range and is
     * equal to n_locally_relevant.
     */
    DEAL_II_HOST_DEVICE
    unsigned int n_rows() const;

    /**
     * The total number of nonzero elements of the given sparsity pattern.
     */
    DEAL_II_HOST_DEVICE
    unsigned int n_nonzero_elements() const;

    //@}
    /**
     * @name Properties of a row
     */
    //@{

    /**
     * Return the "stride size" of a given row index, i.e., the distance in
     * memory between two consecutive column entries of the row. The
     * function returns warp_size for all indices in the range [0,
     * n_internal_dofs) and 1 otherwise.
     */
    DEAL_II_HOST_DEVICE
    unsigned int stride_of_row(const unsigned int row) const;

    /**
     * Return a pointer to the array of column indices for the given row,
     * i.e., for a given row index i:
     * ```
     * const unsigned int *js = sparsity_pattern.columns(i);
     * ```
     * is a pointer to the column index j (or column indices *js when SIMD
     * vectorized).
     */
    DEAL_II_HOST_DEVICE
    const unsigned int *columns(const unsigned int row) const;

    /**
     * Return the row length of a given row index.
     */
    DEAL_II_HOST_DEVICE
    unsigned int row_length(const unsigned int row) const;

    /**
     * Given a row index @p row and a column index @p column return the
     * corresponding column index, i.e., the position of the column in the
     * stencil of nonzero row entries.
     */
    DEAL_II_HOST_DEVICE
    unsigned int column_index(const unsigned int row,
                              const unsigned int column) const;

    //@}
    /**
     * @name Offset calculation
     */
    //@{

    /**
     * Given a row index, an index for the column (within [0,
     * row_length(row)), and a component index return the position of the
     * matrix entry in the data array.
     */
    template <unsigned int n_components = 1>
    DEAL_II_HOST_DEVICE unsigned int
    offset(const unsigned int row,
           const unsigned int column_index,
           const unsigned int component = 0) const;

    /**
     * Specialized version of the function above that computes the offset
     * only for the internal part and pointing to component 0. This variant
     * avoids a number of index computations and an if statement.
     *
     * @note The stride between two components of the returned offset is
     * warp_size. Furthermore, the matrix entries of the rows row, row + 1,
     * ..., row + k of the same warp are stored consecutively starting at
     * the returned offset.
     *
     * @pre row must be within the internal index range.
     */
    template <unsigned int n_components = 1>
    DEAL_II_HOST_DEVICE unsigned int
    offset_internal(const unsigned int row,
                    const unsigned int column_index) const;

    /**
     * Given a row index, an index for the column (within [0,
     * row_length(row)), and a component index return the position of the
     * *transposed* matrix entry in the data array.
     */
    template <unsigned int n_components = 1>
    DEAL_II_HOST_DEVICE unsigned int
    transposed_offset(const unsigned int row,
                      const unsigned int column_index,
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
                               const unsigned int column_index) const;

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

    //@}
    /**
     * @name Internal fields, methods, and friends
     */
    //@{

  private:
    unsigned int n_internal_dofs_;
    unsigned int n_locally_owned_dofs_;

    using KokkosSpace = typename MemorySpace::kokkos_space;
    Kokkos::View<const unsigned int *, KokkosSpace> row_starts_;
    Kokkos::View<const unsigned int *, KokkosSpace> column_indices_;
    Kokkos::View<const unsigned int *, KokkosSpace> indices_transposed_;
    //@}
  };


#ifndef DOXYGEN
  /*
   * -------------------------------------------------------------------------
   * Inline function definitions
   * -------------------------------------------------------------------------
   */


  template <int warp_size>
  template <typename MemorySpace>
  SparsityPatternView<warp_size, MemorySpace>
  SparsityPattern<warp_size>::view() const
  {
    this->template prepare_read_access<MemorySpace>();

    return SparsityPatternView<warp_size, MemorySpace>(*this);
  }


  template <int warp_size>
  template <typename MemorySpace>
  void SparsityPattern<warp_size>::allocate_storage() const
  {
    using HostSpace = dealii::MemorySpace::Host;
    using Aligned = Kokkos::MemoryTraits<Kokkos::Aligned>;

    /* Note: We allocate without initializing. */

    if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
      row_starts_host_ = Kokkos::View<unsigned int *, KokkosHost, Aligned>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "sparsity_pattern_row_starts"),
          row_starts_default_.extent(0));

      column_indices_host_ = Kokkos::View<unsigned int *, KokkosHost, Aligned>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "sparsity_pattern_column_indices"),
          column_indices_default_.extent(0));

      indices_transposed_host_ =
          Kokkos::View<unsigned int *, KokkosHost, Aligned>(
              Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                 "sparsity_pattern_indices_transposed"),
              indices_transposed_default_.extent(0));

    } else {
      row_starts_default_ = Kokkos::View<unsigned int *, KokkosDefault>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "sparsity_pattern_row_starts"),
          row_starts_host_.extent(0));

      column_indices_default_ = Kokkos::View<unsigned int *, KokkosDefault>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "sparsity_pattern_column_indices"),
          column_indices_host_.extent(0));

      indices_transposed_default_ = Kokkos::View<unsigned int *, KokkosDefault>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "sparsity_pattern_indices_transposed"),
          indices_transposed_host_.extent(0));
    }
  }


  template <int warp_size>
  template <typename To, typename From>
  void SparsityPattern<warp_size>::deep_copy_storage() const
  {
    using HostSpace = dealii::MemorySpace::Host;

    if constexpr (std::is_same_v<To, HostSpace>) {
      Kokkos::deep_copy(/*dst*/ row_starts_host_, /*src*/ row_starts_default_);
      Kokkos::deep_copy(/*dst*/ column_indices_host_,
                        /*src*/ column_indices_default_);
      Kokkos::deep_copy(/*dst*/ indices_transposed_host_,
                        /*src*/ indices_transposed_default_);
    } else {
      Kokkos::deep_copy(/*dst*/ row_starts_default_, /*src*/ row_starts_host_);
      Kokkos::deep_copy(/*dst*/ column_indices_default_,
                        /*src*/ column_indices_host_);
      Kokkos::deep_copy(/*dst*/ indices_transposed_default_,
                        /*src*/ indices_transposed_host_);
    }
  }


  template <int warp_size>
  template <typename MemorySpace>
  void SparsityPattern<warp_size>::deallocate_storage()
  {
    using HostSpace = dealii::MemorySpace::Host;

    if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
      row_starts_host_ = {};
      column_indices_host_ = {};
      indices_transposed_host_ = {};

    } else {
      row_starts_default_ = {};
      column_indices_default_ = {};
      indices_transposed_default_ = {};
    }
  }


  template <int warp_size, typename MemorySpace>
  SparsityPatternView<warp_size, MemorySpace>::SparsityPatternView(
      const SparsityPattern<warp_size> &sparsity_pattern)
  {
    reinit(sparsity_pattern);
  }


  template <int warp_size, typename MemorySpace>
  void SparsityPatternView<warp_size, MemorySpace>::reinit(
      const SparsityPattern<warp_size> &sparsity_pattern)
  {
    n_internal_dofs_ = sparsity_pattern.n_internal_dofs_;
    n_locally_owned_dofs_ = sparsity_pattern.n_locally_owned_dofs_;

    using HostSpace = dealii::MemorySpace::Host;
    using DefaultSpace = dealii::MemorySpace::Default;

    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    /*
     * Note: If the host and default memory spaces coincide all views
     * reference the host storage.
     */
    if constexpr (have_separate_memory_spaces &&
                  !std::is_same_v<MemorySpace, HostSpace>) {
      row_starts_ = sparsity_pattern.row_starts_default_;
      column_indices_ = sparsity_pattern.column_indices_default_;
      indices_transposed_ = sparsity_pattern.indices_transposed_default_;
    } else {
      row_starts_ = sparsity_pattern.row_starts_host_;
      column_indices_ = sparsity_pattern.column_indices_host_;
      indices_transposed_ = sparsity_pattern.indices_transposed_host_;
    }
  }


  template <int warp_size, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<warp_size, MemorySpace>::n_internal_dofs() const
  {
    return n_internal_dofs_;
  }


  template <int warp_size, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<warp_size, MemorySpace>::n_locally_owned_dofs() const
  {
    return n_locally_owned_dofs_;
  }


  template <int warp_size, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<warp_size, MemorySpace>::n_rows() const
  {
    Assert(row_starts_.size() > 0, dealii::ExcNotInitialized());

    return row_starts_.size() - 1;
  }


  template <int warp_size, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<warp_size, MemorySpace>::n_nonzero_elements() const
  {
    Assert(row_starts_.size() > 0, dealii::ExcNotInitialized());

    return row_starts_(row_starts_.size() - 1);
  }


  template <int warp_size, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<warp_size, MemorySpace>::stride_of_row(
      const unsigned int row) const
  {
    AssertIndexRange(row, n_rows());

    if (row < n_internal_dofs_)
      return warp_size;
    else
      return 1;
  }


  template <int warp_size, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE const unsigned int *
  SparsityPatternView<warp_size, MemorySpace>::columns(
      const unsigned int row) const
  {
    AssertIndexRange(row, n_rows());

    if (row < n_internal_dofs_)
      return column_indices_.data() + row_starts_(row / warp_size) +
             row % warp_size;
    else
      return column_indices_.data() + row_starts_(row);
  }


  template <int warp_size, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<warp_size, MemorySpace>::row_length(
      const unsigned int row) const
  {
    AssertIndexRange(row, n_rows());

    if (row < n_internal_dofs_) {
      const unsigned int warp = row / warp_size;
      return (row_starts_(warp + 1) - row_starts_(warp)) / warp_size;
    } else {
      return row_starts_(row + 1) - row_starts_(row);
    }
  }


  template <int warp_size, typename MemorySpace>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<warp_size, MemorySpace>::column_index(
      const unsigned int row, const unsigned int column) const
  {
    const auto &row_length = this->row_length(row);
    const auto &stride_size = this->stride_of_row(row);

    // FIXME: with C++23 use std::views::stride and binary search instead...

    const unsigned int *js = columns(row);
    for (unsigned int k = 0; k < row_length; ++k)
      if (js[k * stride_size] == column)
        return k;

    Assert(false, dealii::ExcMessage("Column index not found in given row"));
    return -1;
  }


  template <int warp_size, typename MemorySpace>
  template <unsigned int n_components>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<warp_size, MemorySpace>::offset(
      const unsigned int row,
      const unsigned int column_index,
      const unsigned int comp) const
  {
    AssertIndexRange(row, n_rows());
    AssertIndexRange(column_index, row_length(row));
    AssertIndexRange(comp, n_components);

    const unsigned int warp = row / warp_size;
    const unsigned int lane = row % warp_size;

    if (row < n_internal_dofs_) {
      const unsigned int scalar_offset =
          row_starts_(warp) + column_index * warp_size;
      return scalar_offset * n_components + comp * warp_size + lane;

    } else {
      const unsigned int scalar_offset = row_starts_(row) + column_index;

      return scalar_offset * n_components + comp;
    }
  }


  template <int warp_size, typename MemorySpace>
  template <unsigned int n_components>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<warp_size, MemorySpace>::transposed_offset(
      const unsigned int row,
      const unsigned int column_index,
      const unsigned int component) const
  {
    AssertIndexRange(row, n_rows());
    AssertIndexRange(column_index, row_length(row));
    AssertIndexRange(component, n_components);

    // Compute the transposed index from the (scalar) numbering stored in
    // the sparsity pattern...
    const unsigned int scalar_offset = offset(row, column_index);
    const unsigned int transposed_scalar_offset =
        indices_transposed_(scalar_offset);

    // ... and reconstruct the proper index for a view with n_components:
    const unsigned int j = column_indices_(scalar_offset);

    unsigned int transposed_offset = transposed_scalar_offset;
    if constexpr (n_components > 1) {
      if (j < n_internal_dofs_) {
        transposed_offset = //
            transposed_offset / warp_size * warp_size * n_components +
            transposed_offset % warp_size;
        return transposed_offset + component * warp_size;

      } else {

        transposed_offset *= n_components;
        return transposed_offset + component;
      }

    } else {

      return transposed_offset;
    }
  }


  template <int warp_size, typename MemorySpace>
  template <unsigned int n_components>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<warp_size, MemorySpace>::offset_internal(
      const unsigned int row, const unsigned int column_index) const
  {
    AssertIndexRange(row, n_rows());
    AssertIndexRange(column_index, row_length(row));
    AssertIndexRange(row, n_internal_dofs_);

    const unsigned int warp = row / warp_size;
    const unsigned int lane = row % warp_size;

    const unsigned int scalar_offset =
        row_starts_(warp) + column_index * warp_size;

    return scalar_offset * n_components + lane;
  }


  template <int warp_size, typename MemorySpace>
  template <unsigned int n_components>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE const unsigned int *
  SparsityPatternView<warp_size, MemorySpace>::transposed_offset_internal(
      const unsigned int row, const unsigned int column_index) const
  {
    static_assert(n_components == 1,
                  "Vectorized transposed access to multiple components is not "
                  "yet implemented.");
    AssertIndexRange(row, row_starts_.size() - 1);
    AssertIndexRange(column_index, row_length(row));
    AssertIndexRange(row, n_internal_dofs_);

    const unsigned int warp = row / warp_size;
    const unsigned int lane = row % warp_size;

    const unsigned int scalar_offset =
        row_starts_(warp) + column_index * warp_size;

    // n_components == 1
    return indices_transposed_.data() + scalar_offset + lane;
  }


  template <int warp_size, typename MemorySpace>
  template <unsigned int n_components>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
  SparsityPatternView<warp_size, MemorySpace>::ghost_offset() const
  {
    const auto scalar_offset = row_starts_(n_locally_owned_dofs_);
    return scalar_offset * n_components;
  }


#endif
} // namespace ryujin
