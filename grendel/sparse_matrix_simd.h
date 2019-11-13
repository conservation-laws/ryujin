#ifndef SPARSE_MATRIX_SIMD
#define SPARSE_MATRIX_SIMD

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/lac/sparsity_pattern.h>

#include "helper.h"

namespace grendel
{
  template <int simd_length, typename Number, int n_components = 1>
  class SparseMatrixSIMD;

  template <int simd_length>
  class SparsityPatternSIMD
  {
  public:
    SparsityPatternSIMD()
        : n_internal_dofs(0)
        , row_starts(1)
        , mpi_communicator(MPI_COMM_SELF)
    {
    }

    SparsityPatternSIMD(const unsigned int n_internal_dofs,
                        const dealii::SparsityPattern &sparsity,
                        const dealii::Utilities::MPI::Partitioner &partitioner)
        : n_internal_dofs(0)
        , mpi_communicator(MPI_COMM_SELF)
    {
      reinit(n_internal_dofs, sparsity, partitioner);
    }

    void reinit(const unsigned int n_internal_dofs,
                const dealii::SparsityPattern &sparsity,
                const dealii::Utilities::MPI::Partitioner &partitioner)
    {
      this->n_internal_dofs = n_internal_dofs;

      Assert(n_internal_dofs <= sparsity.n_rows(), dealii::ExcInternalError());
      row_starts.resize_fast(sparsity.n_rows() + 1);
      column_indices.resize_fast(sparsity.n_nonzero_elements());
      column_indices_transposed.resize_fast(sparsity.n_nonzero_elements());
      row_starts[0] = 0;
      Assert(n_internal_dofs % simd_length == 0, dealii::ExcInternalError());
      unsigned int *col_ptr = column_indices.data();
      unsigned int *transposed_ptr = column_indices_transposed.data();
      for (unsigned int i = 0; i < n_internal_dofs; i += simd_length) {
        auto jts = generate_iterators<simd_length>(
            [&](auto k) { return sparsity.begin(i + k); });

        for (; jts[0] != sparsity.end(i); increment_iterators(jts))
          for (unsigned int k = 0; k < simd_length; ++k) {
            *col_ptr++ = jts[k]->column();
            *transposed_ptr++ = sparsity.row_position(jts[k]->column(), i + k);
          }

        row_starts[i / simd_length + 1] = col_ptr - column_indices.data();
      }
      row_starts[n_internal_dofs] = row_starts[n_internal_dofs / simd_length];

      for (unsigned int i = n_internal_dofs; i < sparsity.n_rows(); ++i) {
        for (auto j = sparsity.begin(i); j != sparsity.end(i); ++j) {
          *col_ptr++ = j->column();
          *transposed_ptr++ = sparsity.row_position(j->column(), i);
        }
        row_starts[i + 1] = col_ptr - column_indices.data();
      }

      mpi_communicator = partitioner.get_mpi_communicator();

      Assert(col_ptr == column_indices.end(), dealii::ExcInternalError());

      n_locally_owned_dofs = partitioner.local_size();
      Assert(n_internal_dofs <= n_locally_owned_dofs,
             dealii::ExcInternalError());
      Assert(n_locally_owned_dofs <= sparsity.n_rows(),
             dealii::ExcInternalError());

      /* compute the data exchange pattern */

      if (sparsity.n_rows() > n_locally_owned_dofs) {

        /* stage 1: the processors that are owning the ghosts are the same as
           in the partitioner of the index range */
        auto vec_gt = partitioner.ghost_targets().begin();
        receive_targets.resize(partitioner.ghost_targets().size());

        /* stage 2: remember which range of indices belongs to which
           processor */
        std::vector<unsigned int> ghost_ranges(
            partitioner.ghost_targets().size() + 1);
        ghost_ranges[0] = n_locally_owned_dofs;
        for (unsigned int p = 0; p < receive_targets.size(); ++p) {
          receive_targets[p].first = partitioner.ghost_targets()[p].first;
          ghost_ranges[p + 1] =
              ghost_ranges[p] + partitioner.ghost_targets()[p].second;
        }

        std::vector<unsigned int> import_indices_part;
        for (auto i : partitioner.import_indices())
          for (unsigned int j = i.first; j < i.second; ++j)
            import_indices_part.push_back(j);

        /* Collect indices to be sent. these consist of the diagonal as well
           as the part of columns of the given range. Note that this assumes
           that the sparsity pattern only contains those entries in ghosted
           rows which have a corresponding transpose entry in the owned rows,
           which is the case by construction of the sparsity pattern.
         */
        AssertDimension(import_indices_part.size(),
                        partitioner.n_import_indices());
        indices_to_be_sent.clear();
        send_targets.resize(partitioner.import_targets().size());
        auto idx = import_indices_part.begin();
        for (unsigned int p = 0; p < partitioner.import_targets().size(); ++p) {
          for (unsigned int c = 0; c < partitioner.import_targets()[p].second;
               ++c, ++idx) {
            const unsigned int row = *idx;
            indices_to_be_sent.push_back(row < n_internal_dofs
                                             ? row_starts[row / simd_length] +
                                                   row % simd_length
                                             : row_starts[row]);
            for (auto jt = ++sparsity.begin(row); jt != sparsity.end(row); ++jt)
              if (jt->column() >= ghost_ranges[p] &&
                  jt->column() < ghost_ranges[p + 1])
                indices_to_be_sent.push_back(
                    row < n_internal_dofs
                        ? row_starts[row / simd_length] + row % simd_length +
                              (jt - sparsity.begin(row)) * simd_length
                        : row_starts[row] + (jt - sparsity.begin(row)));
          }
          send_targets[p].first = partitioner.import_targets()[p].first;
          send_targets[p].second = indices_to_be_sent.size();
        }

        /* Count how many dofs to receive and the various buffers to set up
           the MPI communication.
         */
        std::size_t receive_counter = 0;
        unsigned int loc_count = 0;
        for (unsigned int i = n_locally_owned_dofs; i < sparsity.n_rows();
             ++i) {
          receive_counter += sparsity.row_length(i);
          ++loc_count;
          if (loc_count == vec_gt->second) {
            receive_targets[vec_gt - partitioner.ghost_targets().begin()]
                .second = receive_counter;
            loc_count = 0;
            ++vec_gt;
          }
        }

        Assert(vec_gt == partitioner.ghost_targets().end(),
               dealii::ExcInternalError());
      }

      mpi_communicator = partitioner.get_mpi_communicator();
    }

    unsigned int stride_of_row(const unsigned int row) const
    {
      AssertIndexRange(row, row_starts.size());
      if (row < n_internal_dofs)
        return simd_length;
      else
        return 1;
    }

    const unsigned int *columns(const unsigned int row) const
    {
      AssertIndexRange(row, row_starts.size() - 1);
      if (row < n_internal_dofs)
        return column_indices.data() + row_starts[row / simd_length];
      else
        return column_indices.data() + row_starts[row];
    }

    unsigned int row_length(const unsigned int row) const
    {
      AssertIndexRange(row, row_starts.size() - 1);
      if (row < n_internal_dofs) {
        const unsigned int simd_row = row / simd_length;
        return (row_starts[simd_row + 1] - row_starts[simd_row]) / simd_length;
      } else
        return row_starts[row + 1] - row_starts[row];
    }

    unsigned int n_rows() const
    {
      return row_starts.size() - 1;
    }

    std::size_t n_nonzero_elements() const
    {
      Assert(row_starts.size() > 0, dealii::ExcNotInitialized());
      return row_starts.back();
    }

  private:
    unsigned int n_internal_dofs;
    unsigned int n_locally_owned_dofs;
    dealii::AlignedVector<std::size_t> row_starts;
    dealii::AlignedVector<unsigned int> column_indices;
    dealii::AlignedVector<unsigned int> column_indices_transposed;

    dealii::AlignedVector<std::size_t> indices_to_be_sent;
    std::vector<std::pair<unsigned int, unsigned int>> send_targets;
    std::vector<std::pair<unsigned int, unsigned int>> receive_targets;
    MPI_Comm mpi_communicator;

    template <int, typename, int>
    friend class SparseMatrixSIMD;
  };


  /**
   * The data layout of this class in the vectorized region is an
   * array-of-struct-of-array type as follows: At the innermost
   * 'array' level, we group data from simd_length rows contiguously
   * in memory, using a given column block as determined by the
   * sparsity pattern. Next come the different components in case we
   * have a multi-component matrix, i.e., the 'struct' level groups
   * the components next to the inner array of row data. Finally, the
   * outer array aligns the different components in a CSR format,
   * i.e., row-by-row (or row-chunk-per-row-chunk) and along columns,
   * following the sparsity pattern.
   */
  template <int simd_length, typename Number, int n_components>
  class SparseMatrixSIMD
  {
  public:
    SparseMatrixSIMD()
        : sparsity(nullptr)
    {
    }

    SparseMatrixSIMD(const SparsityPatternSIMD<simd_length> &sparsity)
        : sparsity(&sparsity)
    {
      data.resize(sparsity.n_nonzero_elements() * n_components);
    }

    void reinit(const SparsityPatternSIMD<simd_length> &sparsity)
    {
      this->sparsity = &sparsity;
      data.resize(sparsity.n_nonzero_elements() * n_components);
    }

    DEAL_II_ALWAYS_INLINE
    dealii::VectorizedArray<Number, simd_length>
    get_vectorized_entry(const unsigned int row,
                         const unsigned int position_within_column) const
    {
      static_assert(
          n_components == 1,
          "Vectorized entry only available for single-component case");
      return get_vectorized_tensor(row, position_within_column)[0];
    }

    DEAL_II_ALWAYS_INLINE
    dealii::
        Tensor<1, n_components, dealii::VectorizedArray<Number, simd_length>>
        get_vectorized_tensor(const unsigned int row,
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

    DEAL_II_ALWAYS_INLINE
    void write_vectorized_entry(
        const dealii::VectorizedArray<Number, simd_length> entry,
        const unsigned int row,
        const unsigned int position_within_column,
        const bool do_streaming_store = false)
    {
      static_assert(
          n_components == 1,
          "Vectorized entry only available for single-component case");
      dealii::
          Tensor<1, n_components, dealii::VectorizedArray<Number, simd_length>>
              tensor;
      tensor[0] = entry;
      write_vectorized_tensor(
          tensor, row, position_within_column, do_streaming_store);
    }

    DEAL_II_ALWAYS_INLINE
    void write_vectorized_tensor(
        const dealii::Tensor<1,
                             n_components,
                             dealii::VectorizedArray<Number, simd_length>>
            &entry,
        const unsigned int row,
        const unsigned int position_within_column,
        const bool do_streaming_store = false)
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
    }

    DEAL_II_ALWAYS_INLINE
    Number get_entry(const unsigned int row,
                     const unsigned int position_within_column) const
    {
      static_assert(
          n_components == 1,
          "Single-entry access only available for single-component case");
      return get_tensor(row, position_within_column)[0];
    }

    DEAL_II_ALWAYS_INLINE
    dealii::Tensor<1, n_components, Number>
    get_tensor(const unsigned int row,
               const unsigned int position_within_column) const
    {
      Assert(sparsity != nullptr, dealii::ExcNotInitialized());

      AssertIndexRange(row, sparsity->row_starts.size() - 1);
      AssertIndexRange(position_within_column, sparsity->row_length(row));

      dealii::Tensor<1, n_components, Number> result;
      // go through vectorized part
      if (row < sparsity->n_internal_dofs) {
        const unsigned int simd_row = row / simd_length;
        const unsigned int simd_offset = row % simd_length;
        for (unsigned int d = 0; d < n_components; ++d)
          result[d] = data[(sparsity->row_starts[simd_row] +
                            position_within_column * simd_length) *
                               n_components +
                           d * simd_length + simd_offset];
      }
      // go through standard part
      else
        for (unsigned int d = 0; d < n_components; ++d)
          result[d] =
              data[(sparsity->row_starts[row] + position_within_column) *
                       n_components +
                   d];

      return result;
    }

    void write_entry(const Number entry,
                     const unsigned int row,
                     const unsigned int position_within_column)
    {
      static_assert(
          n_components == 1,
          "Single-entry access only available for single-component case");
      dealii::Tensor<1, n_components, Number> result;
      result[0] = entry;
      write_entry(result, row, position_within_column);
    }

    void write_entry(const dealii::Tensor<1, n_components, Number> &entry,
                     const unsigned int row,
                     const unsigned int position_within_column)
    {
      Assert(sparsity != nullptr, dealii::ExcNotInitialized());

      AssertIndexRange(row, sparsity->row_starts.size() - 1);
      AssertIndexRange(position_within_column, sparsity->row_length(row));

      // go through vectorized part
      if (row < sparsity->n_internal_dofs) {
        const unsigned int simd_row = row / simd_length;
        const unsigned int simd_offset = row % simd_length;
        for (unsigned int d = 0; d < n_components; ++d)
          data[(sparsity->row_starts[simd_row] +
                position_within_column * simd_length) *
                   n_components +
               d * simd_length + simd_offset] = entry[d];
      }
      // go through standard part
      else
        for (unsigned int d = 0; d < n_components; ++d)
          data[(sparsity->row_starts[row] + position_within_column) *
                   n_components +
               d] = entry[d];
    }

    DEAL_II_ALWAYS_INLINE
    dealii::VectorizedArray<Number, simd_length>
    get_vectorized_transposed_entry(
        const unsigned int row, const unsigned int position_within_column) const
    {
      static_assert(
          n_components == 1,
          "Access currently only available for single-component case");
      Assert(sparsity != nullptr, dealii::ExcNotInitialized());

      AssertIndexRange(row, sparsity->row_starts.size() - 1);
      AssertIndexRange(position_within_column, sparsity->row_length(row));
      Assert(row < sparsity->n_internal_dofs,
             dealii::ExcMessage(
                 "Vectorized access only possible in vectorized part"));
      Assert(row % simd_length == 0,
             dealii::ExcMessage(
                 "Access only supported for rows at the SIMD granularity"));

      dealii::VectorizedArray<Number, simd_length> result = {};
      for (unsigned int k = 0; k < simd_length; ++k) {
        const unsigned int col =
            sparsity->column_indices[sparsity->row_starts[row / simd_length] +
                                     position_within_column * simd_length + k];
        if (col < sparsity->n_internal_dofs)
          result[k] = data[sparsity->row_starts[col / simd_length] +
                           sparsity->column_indices_transposed
                                   [sparsity->row_starts[row / simd_length] +
                                    position_within_column * simd_length + k] *
                               simd_length +
                           col % simd_length];
        else
          result[k] = data[sparsity->row_starts[col] +
                           sparsity->column_indices_transposed
                               [sparsity->row_starts[row / simd_length] +
                                position_within_column * simd_length + k]];
      }
      return result;
    }

    DEAL_II_ALWAYS_INLINE
    Number get_transposed_entry(const unsigned int row,
                                const unsigned int position_within_column) const
    {
      static_assert(
          n_components == 1,
          "Access currently only available for single-component case");
      Assert(sparsity != nullptr, dealii::ExcNotInitialized());

      AssertIndexRange(row, sparsity->row_starts.size() - 1);
      AssertIndexRange(position_within_column, sparsity->row_length(row));

      const unsigned int my_rowstart =
          row < sparsity->n_internal_dofs
              ? sparsity->row_starts[row / simd_length] + row % simd_length
              : sparsity->row_starts[row];
      const unsigned int my_rowstride =
          row < sparsity->n_internal_dofs ? simd_length : 1;
      const unsigned int col =
          row < sparsity->n_internal_dofs
              ? sparsity->column_indices[my_rowstart +
                                         position_within_column * simd_length]
              : sparsity->column_indices[my_rowstart + position_within_column];
      if (col < sparsity->n_internal_dofs)
        return data[sparsity->row_starts[col / simd_length] +
                    sparsity->column_indices_transposed[my_rowstart +
                                                        position_within_column *
                                                            my_rowstride] *
                        simd_length +
                    col % simd_length];
      else
        return data[sparsity->row_starts[col] +
                    sparsity->column_indices_transposed
                        [my_rowstart + position_within_column * my_rowstride]];
    }

    DEAL_II_ALWAYS_INLINE
    dealii::Tensor<1, n_components, Number>
    get_transposed_tensor(const unsigned int row,
                          const unsigned int position_within_column) const
    {
      Assert(sparsity != nullptr, dealii::ExcNotInitialized());

      AssertIndexRange(row, sparsity->row_starts.size() - 1);
      AssertIndexRange(position_within_column, sparsity->row_length(row));

      const unsigned int my_rowstart =
          row < sparsity->n_internal_dofs
              ? sparsity->row_starts[row / simd_length] + row % simd_length
              : sparsity->row_starts[row];
      const unsigned int my_rowstride =
          row < sparsity->n_internal_dofs ? simd_length : 1;
      const unsigned int col =
          row < sparsity->n_internal_dofs
              ? sparsity->column_indices[my_rowstart +
                                         position_within_column * simd_length]
              : sparsity->column_indices[my_rowstart + position_within_column];

      const unsigned int position_within_transposed_column =
            sparsity->column_indices_transposed[my_rowstart +
                                                position_within_column *
                                                    my_rowstride];

      return get_tensor(col, position_within_transposed_column);
    }

    void communicate_offproc_entries()
    {
      static_assert(n_components == 1, "Only scalar case implemented");
#ifdef DEAL_II_WITH_MPI

      const std::size_t n_indices = sparsity->indices_to_be_sent.size();
      exchange_buffer.resize_fast(n_indices);

      std::vector<MPI_Request> requests(sparsity->receive_targets.size() +
                                        sparsity->send_targets.size());
      {
        const auto &targets = sparsity->receive_targets;
        for (unsigned int p = 0; p < targets.size(); ++p) {
          const int ierr = MPI_Irecv(
              data.data() +
                  sparsity->row_starts[sparsity->n_locally_owned_dofs] +
                  (p == 0 ? 0 : targets[p - 1].second),
              (targets[p].second - (p == 0 ? 0 : targets[p - 1].second)) *
                  sizeof(Number),
              MPI_BYTE,
              targets[p].first,
              13,
              sparsity->mpi_communicator,
              &requests[p]);
          AssertThrowMPI(ierr);
        }
      }

      GRENDEL_PARALLEL_REGION_BEGIN

      GRENDEL_OMP_FOR
      for (std::size_t c = 0; c < n_indices; ++c)
        exchange_buffer[c] = data[sparsity->indices_to_be_sent[c]];

      GRENDEL_PARALLEL_REGION_END

      {
        const auto &targets = sparsity->send_targets;
        for (unsigned int p = 0; p < targets.size(); ++p) {
          const int ierr = MPI_Isend(
              exchange_buffer.data() + (p == 0 ? 0 : targets[p - 1].second),
              (targets[p].second - (p == 0 ? 0 : targets[p - 1].second)) *
                  sizeof(Number),
              MPI_BYTE,
              targets[p].first,
              13,
              sparsity->mpi_communicator,
              &requests[p + sparsity->receive_targets.size()]);
          AssertThrowMPI(ierr);
        }
      }

      const int ierr =
          MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
      AssertThrowMPI(ierr);
#endif
    }

  private:
    const SparsityPatternSIMD<simd_length> *sparsity;
    dealii::AlignedVector<Number> data;
    dealii::AlignedVector<Number> exchange_buffer;
  };

} // namespace grendel

#endif
