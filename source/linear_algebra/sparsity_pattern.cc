//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#include <ryujin/base/compile_time_options.h>

#include "sparsity_pattern.template.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/parallel.h>
#include <deal.II/base/vectorization.h>

#include <algorithm>
#include <map>

namespace ryujin
{
  void DynamicSparsityPattern::reinit(const dealii::IndexSet &rows)
  {
    rows_ = rows;
    rows_.compress();
    columns_.clear();
    columns_.resize(rows_.n_elements());
  }


  void DynamicSparsityPattern::compress()
  {
    dealii::parallel::apply_to_subranges(
        std::size_t(0),
        columns_.size(),
        [&](const std::size_t begin, const std::size_t end) {
          for (auto i = begin; i < end; ++i) {
            auto &columns = columns_[i];
            std::sort(columns.begin(), columns.end());
            columns.erase(std::unique(columns.begin(), columns.end()),
                          columns.end());
          }
        },
        4096);
  }


  void DynamicSparsityPattern::print(std::ostream &out) const
  {
    for (const auto row : rows_) {
      out << '[' << row;
      for (const auto column : columns(row))
        out << ',' << column;
      out << ']' << std::endl;
    }
  }


  void DynamicSparsityPattern::distribute(const dealii::IndexSet &locally_owned,
                                          const MPI_Comm communicator)
  {
    dealii::IndexSet requested_rows = rows_;
    requested_rows.subtract_set(locally_owned);

    const auto owners = dealii::Utilities::MPI::compute_index_owner(
        locally_owned, requested_rows, communicator);

    /*
     * Send every non-empty row as (row, number of columns, columns...):
     */
    std::map<unsigned int, std::vector<size_type>> outgoing;
    std::size_t k = 0;
    for (const auto row : requested_rows) {
      const auto row_columns = columns(row);
      if (!row_columns.empty()) {
        auto &buffer = outgoing[owners[k]];
        buffer.push_back(row);
        buffer.push_back(row_columns.size());
        buffer.insert(buffer.end(), row_columns.begin(), row_columns.end());
      }
      ++k;
    }

    const auto incoming =
        dealii::Utilities::MPI::some_to_some(communicator, outgoing);

    for (const auto &[rank, buffer] : incoming) {
      (void)rank;
      for (auto it = buffer.begin(); it != buffer.end();) {
        const auto row = *it++;
        const auto n_columns = *it++;
        auto &row_columns = columns_[rows_.index_within_set(row)];
        row_columns.insert(row_columns.end(), it, it + n_columns);
        std::sort(row_columns.begin(), row_columns.end());
        row_columns.erase(std::unique(row_columns.begin(), row_columns.end()),
                          row_columns.end());
        it += n_columns;
      }
    }
  }


  /* instantiations */

  template class SparsityPattern<warp_size>;

#if DEAL_II_VECTORIZATION_WIDTH_IN_BITS > 0
  using VA = dealii::VectorizedArray<double>;
  constexpr auto simd_width = VA::size();
  template class SparsityPattern<simd_width == warp_size ? 1 : simd_width>;
#endif
} // namespace ryujin
