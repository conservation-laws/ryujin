//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2025 by the ryujin authors
//

#pragma once

#include "loop.h"
#include "simd.h"
#include "sparse_matrix.h"

#include <deal.II/base/vectorization.h>
#include <deal.II/lac/sparse_matrix.h>

namespace ryujin
{
  template <typename Number, int n_components, int simd_length>
  SparseMatrix<Number, n_components, simd_length>::SparseMatrix()
      : sparsity(nullptr)
  {
  }


  template <typename Number, int n_components, int simd_length>
  SparseMatrix<Number, n_components, simd_length>::SparseMatrix(
      const SparsityPattern<simd_length> &sparsity)
      : sparsity(&sparsity)
  {
    data.resize(sparsity.n_nonzero_elements() * n_components);
  }


  template <typename Number, int n_components, int simd_length>
  void SparseMatrix<Number, n_components, simd_length>::reinit(
      const SparsityPattern<simd_length> &sparsity)
  {
    this->sparsity = &sparsity;
    data.resize(sparsity.n_nonzero_elements() * n_components);
  }


  template <typename Number, int n_components, int simd_length>
  template <typename SparseMatrix2>
  void SparseMatrix<Number, n_components, simd_length>::read_in(
      const std::array<SparseMatrix2, n_components> &sparse_matrix,
      bool locally_indexed /*= true*/)
  {
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

      const unsigned int row_length = sparsity->row_length(i);
      const unsigned int *js = sparsity->columns(i);

      for (unsigned int col_idx = 0; col_idx < row_length;
           ++col_idx, js += stride_size) {

        dealii::Tensor<1, n_components, T> temp;

        if constexpr (std::is_same_v<T, VectorizedArray>) {
          /* Special access for VectorizedArray: */
          for (unsigned int k = 0; k < simd_length; ++k)
            for (unsigned int d = 0; d < n_components; ++d)
              if (locally_indexed)
                temp[d][k] = sparse_matrix[d](i + k, js[k]);
              else
                temp[d][k] = sparse_matrix[d].el(
                    sparsity->partitioner->local_to_global(i + k),
                    sparsity->partitioner->local_to_global(js[k]));

          write_tensor<T>(temp, i, col_idx, true);

        } else {
          for (unsigned int d = 0; d < n_components; ++d)
            if (locally_indexed)
              temp[d] = sparse_matrix[d](i, js[0]);
            else
              temp[d] = sparse_matrix[d].el(
                  sparsity->partitioner->local_to_global(i),
                  sparsity->partitioner->local_to_global(js[0]));
          write_tensor<T>(temp, i, col_idx);
        }
      }
    };

    cpu_simd_loop<Number>("sparse_matrix_read_in",
                          body,
                          0,
                          sparsity->n_internal_dofs,
                          sparsity->n_locally_owned_dofs);
  }


  template <typename Number, int n_components, int simd_length>
  template <typename SparseMatrix2>
  void SparseMatrix<Number, n_components, simd_length>::read_in(
      const SparseMatrix2 &sparse_matrix, bool locally_indexed /*= true*/)
  {
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

      const unsigned int row_length = sparsity->row_length(i);
      const unsigned int *js = sparsity->columns(i);

      for (unsigned int col_idx = 0; col_idx < row_length;
           ++col_idx, js += stride_size) {

        auto temp = T{};

        if constexpr (std::is_same_v<T, VectorizedArray>) {
          for (unsigned int k = 0; k < simd_length; ++k)
            if (locally_indexed)
              temp[k] = sparse_matrix(i + k, js[k]);
            else
              temp[k] = sparse_matrix.el(
                  sparsity->partitioner->local_to_global(i + k),
                  sparsity->partitioner->local_to_global(js[k]));

          write_entry<T>(temp, i, col_idx, true);

        } else {
          temp = locally_indexed
                     ? sparse_matrix(i, js[0])
                     : sparse_matrix.el(
                           sparsity->partitioner->local_to_global(i),
                           sparsity->partitioner->local_to_global(js[0]));
          write_entry<T>(temp, i, col_idx);
        }
      }
    };

    cpu_simd_loop<Number>("sparse_matrix_read_in",
                          body,
                          0,
                          sparsity->n_internal_dofs,
                          sparsity->n_locally_owned_dofs);
  }

} // namespace ryujin
