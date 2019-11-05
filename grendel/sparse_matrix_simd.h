#ifndef SPARSE_MATRIX_SIMD
#define SPARSE_MATRIX_SIMD

#include <deal.II/base/aligned_vector.h>
#include <deal.II/lac/sparsity_pattern.h>

namespace grendel
{
  template <int simd_length, typename Number, int n_components = 1>
  class SparseMatrixForSIMD;

  template <int simd_length>
  class SparsityPatternForSIMD
  {
  public:
    SparsityPatternForSIMD()
        : n_internal_dofs(0)
        , row_starts(1)
    {
    }

    SparsityPatternForSIMD(const unsigned int n_internal_dofs,
                           const dealii::SparsityPattern &sparsity)
        : n_internal_dofs(0)
    {
      reinit(n_internal_dofs, sparsity);
    }

    void reinit(const unsigned int n_internal_dofs,
                const dealii::SparsityPattern &sparsity)
    {
      this->n_internal_dofs = n_internal_dofs;

      Assert(n_internal_dofs <= sparsity.n_rows(), dealii::ExcInternalError());
      row_starts.resize_fast(sparsity.n_rows() + 1);
      column_indices.resize_fast(sparsity.n_nonzero_elements());
      row_starts[0] = 0;
      Assert(n_internal_dofs % simd_length == 0, dealii::ExcInternalError());
      unsigned int *col_ptr = column_indices.data();
      for (unsigned int i = 0; i < n_internal_dofs; i += simd_length) {
        for (unsigned int v = 1; v < simd_length; ++v)
          row_starts[i + v] = row_starts[i] + v;
        auto jts = generate_iterators<simd_length>(
            [&](auto k) { return sparsity.begin(i + k); });

        for (; jts[0] != sparsity.end(i); increment_iterators(jts))
          for (unsigned int k = 0; k < simd_length; ++k)
            *col_ptr++ = jts[k]->column();

        row_starts[i + simd_length] = col_ptr - column_indices.data();
      }

      for (unsigned int i = n_internal_dofs; i < sparsity.n_rows(); ++i) {
        for (auto j = sparsity.begin(i); j != sparsity.end(i); ++j)
          *col_ptr++ = j->column();
        row_starts[i + 1] = col_ptr - column_indices.data();
      }

      Assert(col_ptr == column_indices.end(), dealii::ExcInternalError());
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
      return column_indices.data() + row_starts[row];
    }

    unsigned int row_length(const unsigned int row) const
    {
      AssertIndexRange(row, row_starts.size() - 1);
      if (row < n_internal_dofs) {
        const unsigned int simd_row = row / simd_length * simd_length;
        return (row_starts[simd_row + simd_length] - row_starts[simd_row]) /
               simd_length;
      } else
        return row_starts[row + 1] - row_starts[row];
    }

    std::size_t n_nonzero_elements() const
    {
      Assert(row_starts.size() > 0, dealii::ExcNotInitialized());
      return row_starts.back();
    }

  private:
    unsigned int n_internal_dofs;
    dealii::AlignedVector<std::size_t> row_starts;
    dealii::AlignedVector<unsigned int> column_indices;
    dealii::AlignedVector<unsigned int> column_indices_transposed;

    template <int, typename, int>
    friend class SparseMatrixForSIMD;
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
  class SparseMatrixForSIMD
  {
  public:
    SparseMatrixForSIMD()
        : sparsity(nullptr)
    {
    }

    SparseMatrixForSIMD(const SparsityPatternForSIMD<simd_length> &sparsity)
        : sparsity(&sparsity)
    {
      data.resize(sparsity.n_nonzero_elements() * n_components);
    }

    void reinit(const SparsityPatternForSIMD<simd_length> &sparsity)
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
      for (unsigned int d = 0; d < n_components; ++d)
        result[d].load(
            data.data() +
            (sparsity->row_starts[row] + position_within_column * simd_length) *
                n_components +
            d * simd_length);
      return result;
    }

    DEAL_II_ALWAYS_INLINE
    void write_vectorized_entry(
        const dealii::VectorizedArray<Number, simd_length> entry,
        const unsigned int row,
        const unsigned int position_within_column,
        const bool do_streaming_store = true)
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
        const bool do_streaming_store = true)
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
      if (do_streaming_store)
        for (unsigned int d = 0; d < n_components; ++d)
          entry[d].streaming_store(data.data() +
                                   (sparsity->row_starts[row] +
                                    position_within_column * simd_length) *
                                       n_components +
                                   d * simd_length);
      else
        for (unsigned int d = 0; d < n_components; ++d)
          entry[d].store(data.data() +
                         (sparsity->row_starts[row] +
                          position_within_column * simd_length) *
                             n_components +
                         d * simd_length);
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
        const unsigned int simd_row = row / simd_length * simd_length;
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
      write_tensor(result, row, position_within_column);
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
        const unsigned int simd_row = row / simd_length * simd_length;
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

  private:
    const SparsityPatternForSIMD<simd_length> *sparsity;
    dealii::AlignedVector<Number> data;
  };

} // namespace grendel

#endif
