//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "loop.h"
#include "simd.h"
#include "sparsity_pattern.h"

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/config.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <type_traits>

namespace ryujin
{
  template <typename Number,
            int n_comp,
            int simd_length,
            typename MemorySpace = dealii::MemorySpace::Host::kokkos_space,
            bool writable = true>
  class SparseMatrixView;

  /**
   * A specialized sparse matrix for efficient vectorized SIMD access.
   *
   * In the vectorized row index region [0, n_internal_dofs) we store data
   * as an array-of-struct-of-array type (see the documentation of class
   * SparsityPattern for details). For the non-vectorized row index
   * region [n_internal_dofs, n_locally_relevant_dofs) we store the matrix in
   * CSR format (equivalent to the static dealii::SparsityPattern).
   */
  template <typename Number,
            int n_comp = 1,
            int simd_length = dealii::VectorizedArray<Number>::size()>
  class SparseMatrix : public SparseMatrixView<Number, n_comp, simd_length>
  {
  public:
    /**
     * Constructor and initialization (in host memory space):
     */
    //@{

    /**
     * Default constructor.
     */
    SparseMatrix();

    /**
     * Constructor taking a SIMD sparsity pattern as an argument.
     */
    SparseMatrix(const SparsityPattern<simd_length> &sparsity);

    /**
     * Reinit function reinitializes the matrix with the given SIMD
     * sparsity pattern. The locally owned and ghost ranges are zeroed.
     */
    void reinit(const SparsityPattern<simd_length> &sparsity);

    /**
     * Read in values from a given vector of (scalar) sparse matrices that
     * describe our (vector valued) matrix entries.
     */
    template <typename SparseMatrix2>
    void read_in(const std::array<SparseMatrix2, n_comp> &sparse_matrix,
                 bool locally_indexed = true);

    /**
     * Variant of above function for a scalar matrix with n_comp == 1.
     */
    template <typename SparseMatrix2>
    void read_in(const SparseMatrix2 &sparse_matrix2,
                 bool locally_indexed = true);

    //@}
    /**
     * Memory space access and synchronization:
     */
    //@{

    /**
     * Return a writable view on the sparse matrix for the selected memory
     * space.
     */
    template <typename MemorySpace = dealii::MemorySpace::Host::kokkos_space>
    SparseMatrixView<Number, n_comp, simd_length, MemorySpace, true> get_view();

    /**
     * Return a read-only view on the sparse matrix for the selected memory
     * space.
     */
    template <typename MemorySpace = dealii::MemorySpace::Host::kokkos_space>
    SparseMatrixView<Number, n_comp, simd_length, MemorySpace, false>
    get_view() const;

    /**
     * Returns true if the templated memory space is the currently active
     * memory space.
     */
    template <typename MemorySpace>
    bool is_active_memory_space() const;

    /**
     * Move internal data from the currently active memory space to the
     * templated memory space.
     */
    template <typename MemorySpace>
    void move_to_memory_space();

    //@}
    /**
     * MPI synchronization.
     */
    //@{

    /**
     * MPI synchronization: Zero out all ghost rows.
     */
    template <typename MemorySpace>
    void zero_out_ghost_rows_on_memory_space();

    /**
     * MPI synchronization: Import all ghost rows from neighboring MPI
     * ranks on the templated memory space.
     */
    template <typename MemorySpace>
    void update_ghost_rows_on_memory_space();

    /**
     * MPI synchronization: Copy the data that has accumulated in the ghost
     * range to the owning processor. This function operates on the
     * templated memory space.
     */
    template <typename MemorySpace>
    void compress_on_memory_space(dealii::VectorOperation::values operation);

  private:
    //@}
    /**
     * @name Internal fields, methods, and friends
     */
    //@{

    const SparsityPattern<simd_length> *sparsity_ = nullptr; // FIXME shared_ptr

    using HostSpace = dealii::MemorySpace::Host::kokkos_space;
    Kokkos::View<Number *, HostSpace> data_host_;
    Kokkos::View<Number *, HostSpace> exchange_buffer_host_;

    using DefaultSpace = dealii::MemorySpace::Default::kokkos_space;
    Kokkos::View<Number *, DefaultSpace> data_default_;
    Kokkos::View<Number *, DefaultSpace> exchange_buffer_default_;

    bool host_space_active_;

    std::vector<MPI_Request> requests_;

    template <typename, int, int, typename, bool>
    friend class SparseMatrixView;

    //@}
  };


  /**
   * This class models a "view" of the sparse matrix that lives in the host
   * or device memory space. It provides a number of methods for reading
   * and writing matrix entries.
   *
   * @note This class is designed to be captured by value in computation
   * loops with access to either the host or device memory space. As such
   * we do not store a reference to the underlying SparsityPattern but
   * rather raw pointers into the corresponding memory. The view is only
   * valid as long as the underlying SparsityPattern object is not
   * modified.
   */
  template <typename Number,
            int n_comp,
            int simd_length,
            typename MemorySpace,
            bool writable>
  class SparseMatrixView
  {
  public:
    SparseMatrixView() = default;

    SparseMatrixView(SparseMatrix<Number, n_comp, simd_length> &sparse_matrix)
      requires(writable);

    SparseMatrixView(
        const SparseMatrix<Number, n_comp, simd_length> &sparse_matrix)
      requires(!writable);

    template <typename SparseMatrix>
    void reinit(SparseMatrix &sparse_matrix)
      requires(writable != std::is_const_v<SparseMatrix>);

    /* Get scalar or tensor-valued entry: */

    /**
     * Return the (scalar) entry indexed by @p row and @p
     * position_within_column.
     *
     * @note If the template argument @a Number2
     * is a vectorized array a specialized, faster access will be performed.
     * In this case the index @p row must be within the interval
     * [0, n_internal_dofs) and must be divisible by simd_length.
     *
     * @note This function is only available if `n_comp` is equal to 1.
     */
    template <typename Number2 = Number>
    DEAL_II_HOST_DEVICE Number2
    read_entry(const unsigned int row,
               const unsigned int position_within_column) const;

    /**
     * Return the tensor-valued entry indexed by @p row and
     * @p position_within_column. This function performs the same operation
     * as read_entry() except that it always returns the entry as a tensor
     * (even if it is effectively a scalar entry).
     *
     * @note If the template argument @a Number2
     * is a vectorized array a specialized, faster access will be performed.
     * In this case the index @p row must be within the interval
     * [0, n_internal_dofs) and must be divisible by simd_length.
     */
    template <typename Number2 = Number,
              typename Tensor = dealii::Tensor<1, n_comp, Number2>>
    DEAL_II_HOST_DEVICE Tensor
    read_tensor(const unsigned int row,
                const unsigned int position_within_column) const;

    /* Get transposed scalar or tensor-valued entry: */

    /**
     * Return the transposed (sclar) entry indexed by @p row and
     * @p position_within_column.
     *
     * @note If the template argument @a Number2
     * is a vectorized array a specialized, faster access will be performed.
     * In this case the index @p row must be within the interval
     * [0, n_internal_dofs) and must be divisible by simd_length.
     *
     * @note This function is only available if `n_comp` is equal to 1.
     */
    template <typename Number2 = Number>
    DEAL_II_HOST_DEVICE Number2
    read_transposed_entry(const unsigned int row,
                          const unsigned int position_within_column) const;

    /**
     * Return the transposed tensor-valued entry indexed by @p row and
     * @a position_within_column. This function performs the same operation
     * as read_entry() except that it always returns the entry as a tensor
     * (even if it is effectively a scalar entry).
     *
     * @note If the template argument @a Number2
     * is a vectorized array a specialized, faster access will be performed.
     * In this case the index @p row must be within the interval
     * [0, n_internal_dofs) and must be divisible by simd_length.
     */
    template <typename Number2 = Number,
              typename Tensor = dealii::Tensor<1, n_comp, Number2>>
    DEAL_II_HOST_DEVICE Tensor
    read_transposed_tensor(const unsigned int row,
                           const unsigned int position_within_column) const;

    /* Write scalar or tensor entry: */

    /**
     * Write a (scalar-valued) @p entry to the matrix indexed by @p row
     * and @p position_within_column.
     *
     * @note If the template argument @a Number2
     * is a vectorized array a specialized, faster access will be performed.
     * In this case the index @p row must be within the interval
     * [0, n_internal_dofs) and must be divisible by simd_length.
     *
     * @note This function is only available if `n_comp` is equal to 1.
     */
    template <typename Number2 = Number>
    DEAL_II_HOST_DEVICE void
    write_entry(const Number2 entry,
                const unsigned int row,
                const unsigned int position_within_column,
                const bool do_streaming_store = false) const
      requires(writable);

    /**
     * Write a tensor-valued @p entry to the matrix indexed by @p row
     * and @p position_within_column.
     *
     * @note If the template argument @a Number2
     * is a vectorized array a specialized, faster access will be performed.
     * In this case the index @p row must be within the interval
     * [0, n_internal_dofs) and must be divisible by simd_length.
     */
    template <typename Number2 = Number,
              typename Tensor = dealii::Tensor<1, n_comp, Number2>>
    DEAL_II_HOST_DEVICE void
    write_tensor(const Tensor &tensor,
                 const unsigned int row,
                 const unsigned int position_within_column,
                 const bool do_streaming_store = false) const
      requires(writable);

    /**
     * Add a (scalar-valued) @p entry to the matrix indexed by @p row
     * and @p position_within_column.
     *
     * @note If the template argument @a Number2
     * is a vectorized array a specialized, faster access will be performed.
     * In this case the index @p row must be within the interval
     * [0, n_internal_dofs) and must be divisible by simd_length.
     *
     * @note This function is only available if `n_comp` is equal to 1.
     */
    template <typename Number2 = Number>
    DEAL_II_HOST_DEVICE void
    add_entry(const Number2 entry,
              const unsigned int row,
              const unsigned int position_within_column) const
      requires(writable);

    /**
     * Add a tensor-valued @p entry to the matrix indexed by @p row and @p
     * position_within_column.
     *
     * @note If the template argument @a Number2
     * is a vectorized array a specialized, faster access will be performed.
     * In this case the index @p row must be within the interval
     * [0, n_internal_dofs) and must be divisible by simd_length.
     */
    template <typename Number2 = Number,
              typename Tensor = dealii::Tensor<1, n_comp, Number2>>
    DEAL_II_HOST_DEVICE void
    add_tensor(const Tensor &tensor,
               const unsigned int row,
               const unsigned int position_within_column) const
      requires(writable);


    //@}
    /**
     * MPI synchronization.
     */
    //@{

    void zero_out_ghost_rows() const
      requires(writable);

    void update_ghost_rows() const
      requires(writable);

    void compress(dealii::VectorOperation::values operation) const
      requires(writable);

    //@}

  private:
    using SM = SparseMatrix<Number, n_comp, simd_length>;
    std::conditional_t<writable, SM *, const SM *> sparse_matrix_;
    SparsityPatternView<simd_length, MemorySpace> sparsity_;
    Kokkos::View<Number *, MemorySpace> data_;
  };


#ifndef DOXYGEN
  /*
   * -------------------------------------------------------------------------
   * Inline function definitions
   * -------------------------------------------------------------------------
   */


  template <typename Number, int n_components, int simd_length>
  SparseMatrix<Number, n_components, simd_length>::SparseMatrix()
      : sparsity_(nullptr)
      , host_space_active_(true)
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
    this->sparsity_ = &sparsity;
    this->host_space_active_ = true;

    using HostSpace = dealii::MemorySpace::Host::kokkos_space;
    using DefaultSpace = dealii::MemorySpace::Default::kokkos_space;
    using Aligned = Kokkos::MemoryTraits<Kokkos::Aligned>;

    data_host_ = Kokkos::View<Number *, HostSpace, Aligned>(
        "sparse_matrix_data", sparsity.n_nonzero_elements() * n_components);

    data_default_ = Kokkos::create_mirror_view(
        typename DefaultSpace::execution_space(), data_host_);

    const std::size_t n_indices = sparsity.entries_to_be_sent().size();

    exchange_buffer_host_ = Kokkos::View<Number *, HostSpace, Aligned>(
        "sparse_matrix_exchange_buffer", n_components * n_indices);

    exchange_buffer_default_ = Kokkos::create_mirror_view(
        typename DefaultSpace::execution_space(), exchange_buffer_host_);

    /* reinitialize the view: */
    SparseMatrixView<Number, n_components, simd_length>::reinit(*this);
  }


  template <typename Number, int n_components, int simd_length>
  template <typename SparseMatrix2>
  void SparseMatrix<Number, n_components, simd_length>::read_in(
      const std::array<SparseMatrix2, n_components> &sparse_matrix,
      bool locally_indexed /*= true*/)
  {
    using HostSpace = dealii::MemorySpace::Host::kokkos_space;
    Assert(is_active_memory_space<HostSpace>(),
           dealii::ExcMessage("The chosen memory space is not active."));

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

      const unsigned int row_length = sparsity_->row_length(i);
      const unsigned int *js = sparsity_->columns(i);

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
                    sparsity_->partitioner()->local_to_global(i + k),
                    sparsity_->partitioner()->local_to_global(js[k]));

          this->template write_tensor<T>(temp, i, col_idx, true);

        } else {
          for (unsigned int d = 0; d < n_components; ++d)
            if (locally_indexed)
              temp[d] = sparse_matrix[d](i, js[0]);
            else
              temp[d] = sparse_matrix[d].el(
                  sparsity_->partitioner()->local_to_global(i),
                  sparsity_->partitioner()->local_to_global(js[0]));
          this->template write_tensor<T>(temp, i, col_idx);
        }
      }
    };

    cpu_simd_loop<Number>("sparse_matrix_read_in",
                          body,
                          0,
                          sparsity_->n_internal_dofs(),
                          sparsity_->n_locally_owned_dofs());
  }


  template <typename Number, int n_components, int simd_length>
  template <typename SparseMatrix2>
  void SparseMatrix<Number, n_components, simd_length>::read_in(
      const SparseMatrix2 &sparse_matrix, bool locally_indexed /*= true*/)
  {
    using HostSpace = dealii::MemorySpace::Host::kokkos_space;
    Assert(is_active_memory_space<HostSpace>(),
           dealii::ExcMessage("The chosen memory space is not active."));

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

      const unsigned int row_length = sparsity_->row_length(i);
      const unsigned int *js = sparsity_->columns(i);

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
                  sparsity_->partitioner()->local_to_global(i + k),
                  sparsity_->partitioner()->local_to_global(js[k]));

          this->template write_entry<T>(temp, i, col_idx, true);

        } else {
          temp = locally_indexed
                     ? sparse_matrix(i, js[0])
                     : sparse_matrix.el(
                           sparsity_->partitioner()->local_to_global(i),
                           sparsity_->partitioner()->local_to_global(js[0]));
          this->template write_entry<T>(temp, i, col_idx);
        }
      }
    };

    cpu_simd_loop<Number>("sparse_matrix_read_in",
                          body,
                          0,
                          sparsity_->n_internal_dofs(),
                          sparsity_->n_locally_owned_dofs());
  }


  template <typename Number, int n_comp, int simd_length>
  template <typename MemorySpace>
  SparseMatrixView<Number, n_comp, simd_length, MemorySpace, true>
  SparseMatrix<Number, n_comp, simd_length>::get_view()
  {
    Assert(is_active_memory_space<MemorySpace>(),
           dealii::ExcMessage("The chosen memory space is not active."));

    return SparseMatrixView<Number, n_comp, simd_length, MemorySpace, true>(
        *this);
  }


  template <typename Number, int n_comp, int simd_length>
  template <typename MemorySpace>
  SparseMatrixView<Number, n_comp, simd_length, MemorySpace, false>
  SparseMatrix<Number, n_comp, simd_length>::get_view() const
  {
    Assert(is_active_memory_space<MemorySpace>(),
           dealii::ExcMessage("The chosen memory space is not active."));

    return SparseMatrixView<Number, n_comp, simd_length, MemorySpace, false>(
        *this);
  }


  template <typename Number, int n_components, int simd_length>
  template <typename MemorySpace>
  bool SparseMatrix<Number, n_components, simd_length>::is_active_memory_space()
      const
  {
    using HostSpace = dealii::MemorySpace::Host::kokkos_space;
    using DefaultSpace = dealii::MemorySpace::Default::kokkos_space;
    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected Kokkos memory space");

    return host_space_active_ == std::is_same_v<MemorySpace, HostSpace>;
  }


  template <typename Number, int n_components, int simd_length>
  template <typename MemorySpace>
  void SparseMatrix<Number, n_components, simd_length>::move_to_memory_space()
  {
    using HostSpace = dealii::MemorySpace::Host::kokkos_space;
    using DefaultSpace = dealii::MemorySpace::Default::kokkos_space;
    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected Kokkos memory space");

    if (is_active_memory_space<MemorySpace>())
      return;

    if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
      host_space_active_ = true;
      Kokkos::deep_copy(/*dst*/ data_host_, /*src*/ data_default_);
      Kokkos::deep_copy(/*dst*/ exchange_buffer_host_,
                        /*src*/ exchange_buffer_default_);

    } else if constexpr (std::is_same_v<MemorySpace, DefaultSpace>) {
      host_space_active_ = false;
      Kokkos::deep_copy(/*dst*/ data_default_, /*src*/ data_host_);
      Kokkos::deep_copy(/*dst*/ exchange_buffer_default_,
                        /*src*/ exchange_buffer_host_);
    }
  }


  template <typename Number, int n_components, int simd_length>
  template <typename MemorySpace>
  void SparseMatrix<Number, n_components, simd_length>::
      zero_out_ghost_rows_on_memory_space()
  {
    using HostSpace = dealii::MemorySpace::Host::kokkos_space;
    using DefaultSpace = dealii::MemorySpace::Default::kokkos_space;

    Assert(is_active_memory_space<MemorySpace>(),
           dealii::ExcMessage("The chosen memory space is not active."));

    AssertThrow((std::is_same_v<MemorySpace, HostSpace>),
                dealii::ExcNotImplemented());

    const auto ghost_offset = sparsity_->template ghost_offset<n_components>();
    const auto end_offset = sparsity_->n_nonzero_elements() * n_components;
    std::fill(data_host_.data() + ghost_offset,
              data_host_.data() + end_offset,
              Number{});
  }


  template <typename Number, int n_components, int simd_length>
  template <typename MemorySpace>
  void SparseMatrix<Number, n_components, simd_length>::
      update_ghost_rows_on_memory_space()
  {
    using HostSpace = dealii::MemorySpace::Host::kokkos_space;
    using DefaultSpace = dealii::MemorySpace::Default::kokkos_space;

    AssertThrow((std::is_same_v<MemorySpace, HostSpace>),
                dealii::ExcNotImplemented());

    Assert(is_active_memory_space<MemorySpace>(),
           dealii::ExcMessage("The chosen memory space is not active."));

    const auto &receive_targets = sparsity_->receive_targets();
    const auto &send_targets = sparsity_->send_targets();
    const auto &entries_to_be_sent = sparsity_->entries_to_be_sent();

    const unsigned int mpi_tag =
        dealii::Utilities::MPI::internal::Tags::partitioner_export_start + 0;
    Assert(mpi_tag <=
               dealii::Utilities::MPI::internal::Tags::partitioner_export_end,
           dealii::ExcInternalError());

    const unsigned int n_requests =
        receive_targets.size() + send_targets.size();
    std::vector<MPI_Request> requests(n_requests);

    const auto ghost_offset = sparsity_->template ghost_offset<n_components>();

    for (unsigned int p = 0; p < receive_targets.size(); ++p) {
      const auto receive_offset =
          n_components * (p == 0 ? 0 : receive_targets[p - 1].second);
      const auto receive_size =
          (receive_targets[p].second * n_components - receive_offset);

      const int ierr =
          MPI_Irecv(data_host_.data() + ghost_offset + receive_offset,
                    receive_size,
                    dealii::Utilities::MPI::mpi_type_id_for_type<Number>,
                    receive_targets[p].first,
                    mpi_tag,
                    sparsity_->partitioner()->get_mpi_communicator(),
                    &requests[p]);
      AssertThrowMPI(ierr);
    }

    for (std::size_t c = 0; c < entries_to_be_sent.size(); ++c) {
      const auto &[row, position_within_column] = entries_to_be_sent[c];
      for (unsigned int d = 0; d < n_components; ++d) {
        const auto offset = sparsity_->template offset<n_components>(
            row, position_within_column, d);
        exchange_buffer_host_(n_components * c + d) = data_host_(offset);
      }
    }

    for (unsigned int p = 0; p < send_targets.size(); ++p) {
      const auto send_offset =
          n_components * (p == 0 ? 0 : send_targets[p - 1].second);
      const auto send_size =
          (send_targets[p].second * n_components - send_offset);

      const int ierr =
          MPI_Isend(exchange_buffer_host_.data() + send_offset,
                    send_size,
                    dealii::Utilities::MPI::mpi_type_id_for_type<Number>,
                    send_targets[p].first,
                    mpi_tag,
                    sparsity_->partitioner()->get_mpi_communicator(),
                    &requests[receive_targets.size() + p]);
      AssertThrowMPI(ierr);
    }

    const int ierr =
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    AssertThrowMPI(ierr);
  }


  template <typename Number, int n_components, int simd_length>
  template <typename MemorySpace>
  void
  SparseMatrix<Number, n_components, simd_length>::compress_on_memory_space(
      dealii::VectorOperation::values operation)
  {
    Assert(operation == dealii::VectorOperation::add,
           dealii::ExcNotImplemented());

    using HostSpace = dealii::MemorySpace::Host::kokkos_space;
    using DefaultSpace = dealii::MemorySpace::Default::kokkos_space;

    AssertThrow((std::is_same_v<MemorySpace, HostSpace>),
                dealii::ExcNotImplemented());

    Assert(is_active_memory_space<MemorySpace>(),
           dealii::ExcMessage("The chosen memory space is not active."));

    const auto &receive_targets = sparsity_->receive_targets();
    const auto &send_targets = sparsity_->send_targets();
    const auto &entries_to_be_sent = sparsity_->entries_to_be_sent();

    const unsigned int mpi_tag =
        dealii::Utilities::MPI::internal::Tags::partitioner_export_start + 0;
    Assert(mpi_tag <=
               dealii::Utilities::MPI::internal::Tags::partitioner_export_end,
           dealii::ExcInternalError());

    const unsigned int n_requests =
        receive_targets.size() + send_targets.size();
    std::vector<MPI_Request> requests(n_requests);

    /* Note: For compress() we receive from the "send targets" */
    for (unsigned int p = 0; p < send_targets.size(); ++p) {
      const auto receive_offset =
          n_components * (p == 0 ? 0 : send_targets[p - 1].second);
      const auto receive_size =
          (send_targets[p].second * n_components - receive_offset);

      const int ierr =
          MPI_Irecv(exchange_buffer_host_.data() + receive_offset,
                    receive_size,
                    dealii::Utilities::MPI::mpi_type_id_for_type<Number>,
                    send_targets[p].first,
                    mpi_tag,
                    sparsity_->partitioner()->get_mpi_communicator(),
                    &requests[p]);
      AssertThrowMPI(ierr);
    }

    const auto ghost_offset = sparsity_->template ghost_offset<n_components>();

    /* Note: For compress() we send to the "receive targets" */
    for (unsigned int p = 0; p < receive_targets.size(); ++p) {
      const auto send_offset =
          n_components * (p == 0 ? 0 : receive_targets[p - 1].second);
      const auto send_size =
          (receive_targets[p].second * n_components - send_offset);

      const int ierr =
          MPI_Isend(data_host_.data() + ghost_offset + send_offset,
                    send_size,
                    dealii::Utilities::MPI::mpi_type_id_for_type<Number>,
                    receive_targets[p].first,
                    mpi_tag,
                    sparsity_->partitioner()->get_mpi_communicator(),
                    &requests[send_targets.size() + p]);
      AssertThrowMPI(ierr);
    }

    const int ierr =
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    AssertThrowMPI(ierr);

    /* Add back contributions and clear ghost range: */

    for (std::size_t c = 0; c < entries_to_be_sent.size(); ++c) {
      const auto &[row, position_within_column] = entries_to_be_sent[c];
      for (unsigned int d = 0; d < n_components; ++d) {
        const auto offset = sparsity_->template offset<n_components>(
            row, position_within_column, d);
        data_host_(offset) += exchange_buffer_host_(n_components * c + d);
      }
    }

    zero_out_ghost_rows_on_memory_space<MemorySpace>();
  }


  template <typename Number,
            int n_comp,
            int simd_length,
            typename MemorySpace,
            bool writable>
  SparseMatrixView<Number, n_comp, simd_length, MemorySpace, writable>::
      SparseMatrixView(SparseMatrix<Number, n_comp, simd_length> &sparse_matrix)
    requires(writable)
  {
    reinit(sparse_matrix);
  }


  template <typename Number,
            int n_comp,
            int simd_length,
            typename MemorySpace,
            bool writable>
  SparseMatrixView<Number, n_comp, simd_length, MemorySpace, writable>::
      SparseMatrixView(
          const SparseMatrix<Number, n_comp, simd_length> &sparse_matrix)
    requires(!writable)
  {
    reinit(sparse_matrix);
  }


  template <typename Number,
            int n_comp,
            int simd_length,
            typename MemorySpace,
            bool writable>
  template <typename SparseMatrix>
  void
  SparseMatrixView<Number, n_comp, simd_length, MemorySpace, writable>::reinit(
      SparseMatrix &sparse_matrix)
    requires(writable != std::is_const_v<SparseMatrix>)
  {
    using HostSpace = dealii::MemorySpace::Host::kokkos_space;
    using DefaultSpace = dealii::MemorySpace::Default::kokkos_space;

    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected Kokkos memory space");

    sparse_matrix_ = &sparse_matrix;

    if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
      data_ = sparse_matrix.data_host_;
    } else {
      data_ = sparse_matrix.data_default_;
    }

    sparsity_ = sparse_matrix.sparsity_->template get_view<MemorySpace>();
  }


  template <typename Number,
            int n_comp,
            int simd_length,
            typename MemorySpace,
            bool writable>
  template <typename Number2>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number2
  SparseMatrixView<Number, n_comp, simd_length, MemorySpace, writable>::
      read_entry(const unsigned int row,
                 const unsigned int position_within_column) const
  {
    static_assert(
        n_comp == 1,
        "Attempted to write a scalar value into a tensor-valued matrix entry");

    const auto result = read_tensor<Number2>(row, position_within_column);
    return result[0];
  }


  template <typename Number,
            int n_comp,
            int simd_length,
            typename MemorySpace,
            bool writable>
  template <typename Number2, typename Tensor>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE Tensor
  SparseMatrixView<Number, n_comp, simd_length, MemorySpace, writable>::
      read_tensor(const unsigned int row,
                  const unsigned int position_within_column) const
  {
    static_assert(std::is_same_v<Number2, typename Tensor::value_type>,
                  "type mismatch");

    AssertIndexRange(row, sparsity_.n_rows());
    AssertIndexRange(position_within_column, sparsity_.row_length(row));

    Tensor result;

    using VA = dealii::VectorizedArray<Number>;
    if constexpr (std::is_same_v<VA, Number2>) {
      /*
       * Vectorized fast access. Indices must be in the range
       * [0,n_internal), index must be divisible by simd_length
       */

      Assert(row < sparsity_.n_internal_dofs(),
             dealii::ExcMessage(
                 "Vectorized access only possible in vectorized part"));
      Assert(row % simd_length == 0,
             dealii::ExcMessage(
                 "Access only supported for rows at the SIMD granularity"));

      const Number *load_pos = data_.data();
      load_pos += sparsity_.template offset_internal<n_comp>(
          row, position_within_column);

      for (unsigned int d = 0; d < n_comp; ++d)
        result[d].load(load_pos + d * simd_length);

    } else {
      /*
       * Non-vectorized slow access. Supports all row indices in [0,n_owned):
       */

      for (unsigned int d = 0; d < n_comp; ++d) {
        const auto offset =
            sparsity_.template offset<n_comp>(row, position_within_column, d);
        result[d] = data_(offset);
      }
    }

    return result;
  }


  template <typename Number,
            int n_comp,
            int simd_length,
            typename MemorySpace,
            bool writable>
  template <typename Number2>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number2
  SparseMatrixView<Number, n_comp, simd_length, MemorySpace, writable>::
      read_transposed_entry(const unsigned int row,
                            const unsigned int position_within_column) const
  {
    static_assert(
        n_comp == 1,
        "Attempted to write a scalar value into a tensor-valued matrix entry");

    const auto result =
        read_transposed_tensor<Number2>(row, position_within_column);
    return result[0];
  }


  template <typename Number,
            int n_comp,
            int simd_length,
            typename MemorySpace,
            bool writable>
  template <typename Number2, typename Tensor>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE Tensor
  SparseMatrixView<Number, n_comp, simd_length, MemorySpace, writable>::
      read_transposed_tensor(const unsigned int row,
                             const unsigned int position_within_column) const
  {
    static_assert(std::is_same_v<Number2, typename Tensor::value_type>,
                  "type mismatch");

    AssertIndexRange(row, sparsity_.n_rows());
    AssertIndexRange(position_within_column, sparsity_.row_length(row));

    dealii::Tensor<1, n_comp, Number2> result;

    using VA = dealii::VectorizedArray<Number>;
    if constexpr (std::is_same_v<VA, Number2> && (n_comp == 1)) {
      /*
       * Vectorized fast access. Indices must be in the range
       * [0,n_internal), index must be divisible by simd_length
       */

      Assert(row < sparsity_.n_internal_dofs(),
             dealii::ExcMessage(
                 "Vectorized access only possible in vectorized part"));
      Assert(row % simd_length == 0,
             dealii::ExcMessage(
                 "Access only supported for rows at the SIMD granularity"));

      const auto offsets = sparsity_.template transposed_offset_internal<1>(
          row, position_within_column);
      result[0].gather(data_.data(), offsets);

    } else if constexpr (std::is_same_v<VA, Number2> && (n_comp != 1)) {

      /* not implemented */
      Assert(false,
             dealii::ExcMessage("Vectorized transposed access to multiple "
                                "components is not implemented."));
      __builtin_trap();

    } else {
      /*
       * Non-vectorized slow access. Supports all row indices in [0,n_owned):
       */

      for (unsigned int d = 0; d < n_comp; ++d) {
        const auto offset = sparsity_.template transposed_offset<n_comp>(
            row, position_within_column, d);
        result[d] = data_(offset);
      }
    }

    return result;
  }


  template <typename Number,
            int n_comp,
            int simd_length,
            typename MemorySpace,
            bool writable>
  template <typename Number2>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
  SparseMatrixView<Number, n_comp, simd_length, MemorySpace, writable>::
      write_entry(const Number2 entry,
                  const unsigned int row,
                  const unsigned int position_within_column,
                  const bool do_streaming_store) const
    requires(writable)
  {
    static_assert(
        n_comp == 1,
        "Attempted to write a scalar value into a tensor-valued matrix entry");

    AssertIndexRange(row, sparsity_.n_rows());
    AssertIndexRange(position_within_column, sparsity_.row_length(row));

    dealii::Tensor<1, n_comp, Number2> tensor;
    tensor[0] = entry;

    write_tensor<Number2>(
        tensor, row, position_within_column, do_streaming_store);
  }


  template <typename Number,
            int n_comp,
            int simd_length,
            typename MemorySpace,
            bool writable>
  template <typename Number2, typename Tensor>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
  SparseMatrixView<Number, n_comp, simd_length, MemorySpace, writable>::
      write_tensor(const Tensor &tensor,
                   const unsigned int row,
                   const unsigned int position_within_column,
                   const bool do_streaming_store) const
    requires(writable)
  {
    AssertIndexRange(row, sparsity_.n_rows());
    AssertIndexRange(position_within_column, sparsity_.row_length(row));

    using VA = dealii::VectorizedArray<Number>;
    if constexpr (std::is_same_v<VA, Number2>) {
      /*
       * Vectorized fast access. Indices must be in the range [0,n_internal),
       * index must be divisible by simd_length:
       */

      Assert(row < sparsity_.n_internal_dofs(),
             dealii::ExcMessage(
                 "Vectorized access only possible in vectorized part"));
      Assert(row % simd_length == 0,
             dealii::ExcMessage(
                 "Access only supported for rows at the SIMD granularity"));

      Number *store_pos = data_.data();
      store_pos += sparsity_.template offset_internal<n_comp>(
          row, position_within_column);

      if (do_streaming_store)
        for (unsigned int d = 0; d < n_comp; ++d)
          tensor[d].streaming_store(store_pos + d * simd_length);
      else
        for (unsigned int d = 0; d < n_comp; ++d)
          tensor[d].store(store_pos + d * simd_length);

    } else {
      /*
       * Non-vectorized slow access. Supports all row indices in [0,n_owned):
       */

      for (unsigned int d = 0; d < n_comp; ++d) {
        const auto offset =
            sparsity_.template offset<n_comp>(row, position_within_column, d);
        data_(offset) = tensor[d];
      }
    }
  }


  template <typename Number,
            int n_comp,
            int simd_length,
            typename MemorySpace,
            bool writable>
  template <typename Number2>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
  SparseMatrixView<Number, n_comp, simd_length, MemorySpace, writable>::
      add_entry(const Number2 entry,
                const unsigned int row,
                const unsigned int position_within_column) const
    requires(writable)
  {
    static_assert(
        n_comp == 1,
        "Attempted to write a scalar value into a tensor-valued matrix entry");

    AssertIndexRange(row, sparsity_.n_rows());
    AssertIndexRange(position_within_column, sparsity_.row_length(row));

    dealii::Tensor<1, n_comp, Number2> tensor;
    tensor[0] = entry;

    add_tensor<Number2>(tensor, row, position_within_column);
  }


  template <typename Number,
            int n_comp,
            int simd_length,
            typename MemorySpace,
            bool writable>
  template <typename Number2, typename Tensor>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
  SparseMatrixView<Number, n_comp, simd_length, MemorySpace, writable>::
      add_tensor(const Tensor &tensor,
                 const unsigned int row,
                 const unsigned int position_within_column) const
    requires(writable)
  {
    AssertIndexRange(row, sparsity_.n_rows());
    AssertIndexRange(position_within_column, sparsity_.row_length(row));

    using VA = dealii::VectorizedArray<Number>;
    if constexpr (std::is_same_v<VA, Number2>) {
      /*
       * Vectorized fast access. Indices must be in the range [0,n_internal),
       * index must be divisible by simd_length:
       */

      Assert(row < sparsity_.n_internal_dofs(),
             dealii::ExcMessage(
                 "Vectorized access only possible in vectorized part"));
      Assert(row % simd_length == 0,
             dealii::ExcMessage(
                 "Access only supported for rows at the SIMD granularity"));

      Number *store_pos = data_.data();
      store_pos += sparsity_.template offset_internal<n_comp>(
          row, position_within_column);

      for (unsigned int d = 0; d < n_comp; ++d) {
        auto temp = tensor[d];
        temp.load(store_pos + d * simd_length);
        temp += tensor[d];
        temp.store(store_pos + d * simd_length);
      }

    } else {
      /*
       * Non-vectorized slow access. Supports all row indices in [0,n_owned):
       */

      for (unsigned int d = 0; d < n_comp; ++d) {
        const auto offset =
            sparsity_.template offset<n_comp>(row, position_within_column, d);
        data_(offset) += tensor[d]; /*add*/
        ;
      }
    }
  }


  template <typename Number,
            int n_comp,
            int simd_length,
            typename MemorySpace,
            bool writable>
  void SparseMatrixView<Number, n_comp, simd_length, MemorySpace, writable>::
      zero_out_ghost_rows() const
    requires(writable)
  {
    using HostSpace = dealii::MemorySpace::Host::kokkos_space;
    using DefaultSpace = dealii::MemorySpace::Default::kokkos_space;

    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected Kokkos memory space");

    Assert(sparse_matrix_->template is_active_memory_space<MemorySpace>(),
           dealii::ExcMessage("The chosen memory space is not active."));

    sparse_matrix_->template zero_out_ghost_rows<MemorySpace>();
  }


  template <typename Number,
            int n_comp,
            int simd_length,
            typename MemorySpace,
            bool writable>
  void SparseMatrixView<Number, n_comp, simd_length, MemorySpace, writable>::
      update_ghost_rows() const
    requires(writable)
  {
    using HostSpace = dealii::MemorySpace::Host::kokkos_space;
    using DefaultSpace = dealii::MemorySpace::Default::kokkos_space;

    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected Kokkos memory space");

    Assert(sparse_matrix_->template is_active_memory_space<MemorySpace>(),
           dealii::ExcMessage("The chosen memory space is not active."));

    sparse_matrix_->template update_ghost_rows_on_memory_space<MemorySpace>();
  }


  template <typename Number,
            int n_comp,
            int simd_length,
            typename MemorySpace,
            bool writable>
  void SparseMatrixView<Number, n_comp, simd_length, MemorySpace, writable>::
      compress(dealii::VectorOperation::values operation) const
    requires(writable)
  {
    using HostSpace = dealii::MemorySpace::Host::kokkos_space;
    using DefaultSpace = dealii::MemorySpace::Default::kokkos_space;

    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected Kokkos memory space");

    Assert(sparse_matrix_->template is_active_memory_space<MemorySpace>(),
           dealii::ExcMessage("The chosen memory space is not active."));

    sparse_matrix_->template compress_on_memory_space<MemorySpace>(operation);
  }

#endif
} // namespace ryujin
