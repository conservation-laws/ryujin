#ifndef SPARSE_MATRIX_SIMD_TEMPLATE_H
#define SPARSE_MATRIX_SIMD_TEMPLATE_H

#include "sparse_matrix_simd.h"

#include <deal.II/base/vectorization.h>
#include <deal.II/lac/sparse_matrix.h>

namespace grendel
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
      const dealii::SparsityPattern &sparsity,
      const dealii::Utilities::MPI::Partitioner &partitioner)
      : n_internal_dofs(0)
      , mpi_communicator(MPI_COMM_SELF)
  {
    reinit(n_internal_dofs, sparsity, partitioner);
  }


  template <int simd_length>
  void SparsityPatternSIMD<simd_length>::reinit(
      const unsigned int n_internal_dofs,
      const dealii::SparsityPattern &sparsity,
      const dealii::Utilities::MPI::Partitioner &partitioner)
  {
    this->mpi_communicator = partitioner.get_mpi_communicator();

    this->n_internal_dofs = n_internal_dofs;
    this->n_locally_owned_dofs = partitioner.local_size();

    Assert(n_internal_dofs <= sparsity.n_rows(), dealii::ExcInternalError());
    Assert(n_internal_dofs % simd_length == 0, dealii::ExcInternalError());
    Assert(n_internal_dofs <= n_locally_owned_dofs, dealii::ExcInternalError());
    Assert(n_locally_owned_dofs <= sparsity.n_rows(),
           dealii::ExcInternalError());

    row_starts.resize_fast(sparsity.n_rows() + 1);
    column_indices.resize_fast(sparsity.n_nonzero_elements());
    column_indices_transposed.resize_fast(sparsity.n_nonzero_elements());

    /* Vectorized part: */

    row_starts[0] = 0;

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

    /* Rest: */

    row_starts[n_internal_dofs] = row_starts[n_internal_dofs / simd_length];

    for (unsigned int i = n_internal_dofs; i < sparsity.n_rows(); ++i) {
      for (auto j = sparsity.begin(i); j != sparsity.end(i); ++j) {
        *col_ptr++ = j->column();
        *transposed_ptr++ = sparsity.row_position(j->column(), i);
      }
      row_starts[i + 1] = col_ptr - column_indices.data();
    }

    Assert(col_ptr == column_indices.end(), dealii::ExcInternalError());

    /* Compute the data exchange pattern: */

    if (sparsity.n_rows() > n_locally_owned_dofs) {

      /*
       * Step 1: The processors that are owning the ghosts are the same as
       * in the partitioner of the index range:
       */

      const auto &ghost_targets = partitioner.ghost_targets();

      auto vec_gt = ghost_targets.begin();
      receive_targets.resize(ghost_targets.size());

      /*
       * Step 2: remember which range of indices belongs to which
       * processor:
       */

      std::vector<unsigned int> ghost_ranges(ghost_targets.size() + 1);

      ghost_ranges[0] = n_locally_owned_dofs;
      for (unsigned int p = 0; p < receive_targets.size(); ++p) {
        receive_targets[p].first = ghost_targets[p].first;
        ghost_ranges[p + 1] = ghost_ranges[p] + ghost_targets[p].second;
      }

      std::vector<unsigned int> import_indices_part;
      for (auto i : partitioner.import_indices())
        for (unsigned int j = i.first; j < i.second; ++j)
          import_indices_part.push_back(j);

      /*
       * Step 3: Collect indices to be sent. these consist of the diagonal,
       * as well as the part of columns of the given range. Note that this
       * assumes that the sparsity pattern only contains those entries in
       * ghosted rows which have a corresponding transpose entry in the
       * owned rows; which is the case for our minimized sparsity pattern.
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

      /*
       * Count how many dofs to receive and the various buffers to set up
       * the MPI communication.
       */

      std::size_t receive_counter = 0;
      unsigned int loc_count = 0;
      for (unsigned int i = n_locally_owned_dofs; i < sparsity.n_rows(); ++i) {
        receive_counter += sparsity.row_length(i);
        ++loc_count;
        if (loc_count == vec_gt->second) {
          receive_targets[vec_gt - partitioner.ghost_targets().begin()].second =
              receive_counter;
          loc_count = 0;
          ++vec_gt;
        }
      }

      Assert(vec_gt == partitioner.ghost_targets().end(),
             dealii::ExcInternalError());
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
  void SparseMatrixSIMD<Number, n_components, simd_length>::read_in(
      const std::array<dealii::SparseMatrix<Number>, n_components>
          &sparse_matrix)
  {
    GRENDEL_PARALLEL_REGION_BEGIN

    /*
     * We use the indirect (and slow) access via operator()(i, j) into the
     * sparse matrix we are copying from. This allows for significantly
     * increased flexibility with respect to the sparsity pattern used in
     * the sparse_matrix object.
     */

    GRENDEL_OMP_FOR
    for (unsigned int i = 0; i < sparsity->n_internal_dofs; i += simd_length) {

      const unsigned int row_length = sparsity->row_length(i);

      const unsigned int *js = sparsity->columns(i);
      for (unsigned int col_idx = 0; col_idx < row_length;
           ++col_idx, js += simd_length) {

        dealii::Tensor<1, n_components, VectorizedArray> temp;
        for (unsigned int k = 0; k < simd_length; ++k)
          for (unsigned int d = 0; d < n_components; ++d)
            temp[d][k] = sparse_matrix[d](i + k, js[k]);

        write_vectorized_tensor(temp, i, col_idx, true);
      }
    }

    const auto n_rows = sparsity->n_rows();
    GRENDEL_OMP_FOR
    for (unsigned int i = sparsity->n_internal_dofs; i < n_rows; ++i) {

      const unsigned int row_length = sparsity->row_length(i);
      const unsigned int *js = sparsity->columns(i);
      for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx, ++js) {

        dealii::Tensor<1, n_components, Number> temp;
        for (unsigned int d = 0; d < n_components; ++d)
          temp[d] = sparse_matrix[d](i, js[0]);
        write_tensor(temp, i, col_idx);
      }
    }

    GRENDEL_PARALLEL_REGION_END
  }


  template <typename Number, int n_components, int simd_length>
  void SparseMatrixSIMD<Number, n_components, simd_length>::read_in(
      const dealii::SparseMatrix<Number> &sparse_matrix)
  {
    GRENDEL_PARALLEL_REGION_BEGIN

    /*
     * We use the indirect (and slow) access via operator()(i, j) into the
     * sparse matrix we are copying from. This allows for significantly
     * increased flexibility with respect to the sparsity pattern used in
     * the sparse_matrix object.
     */

    GRENDEL_OMP_FOR
    for (unsigned int i = 0; i < sparsity->n_internal_dofs; i += simd_length) {

      const unsigned int row_length = sparsity->row_length(i);

      const unsigned int *js = sparsity->columns(i);
      for (unsigned int col_idx = 0; col_idx < row_length;
           ++col_idx, js += simd_length) {

        dealii::VectorizedArray<Number, simd_length> temp = {};
        for (unsigned int k = 0; k < simd_length; ++k)
          temp[k] = sparse_matrix(i + k, js[k]);

        write_vectorized_entry(temp, i, col_idx, true);
      }
    }

    const auto n_rows = sparsity->n_rows();
    GRENDEL_OMP_FOR
    for (unsigned int i = sparsity->n_internal_dofs; i < n_rows; ++i) {

      const unsigned int row_length = sparsity->row_length(i);
      const unsigned int *js = sparsity->columns(i);
      for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx, ++js) {

        const auto temp = sparse_matrix(i, js[0]);
        write_entry(temp, i, col_idx);
      }
    }

    GRENDEL_PARALLEL_REGION_END
  }

} // namespace grendel

#endif /* SPARSE_MATRIX_SIMD_TEMPLATE_H */
