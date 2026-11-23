//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/iterator_range.h>
#include <deal.II/base/parallel.h>
#include <deal.II/base/std_cxx20/type_traits.h>
#include <deal.II/base/types.h>
#include <deal.II/base/work_stream.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace ryujin
{
  /**
   * An extension of the colored variant of dealii::WorkStream::run() that
   * operates on chunks of consecutive elements instead of single elements.
   * The result of a run does not depend on the number of threads.
   *
   * @ingroup Utilities
   */
  namespace WorkStream
  {
    /**
     * Split the range [@p begin, @p end) into chunks of @p chunk_size
     * consecutive elements and color the chunks such that no two chunks of
     * a color share a conflict index. The function @p get_conflict_indices
     * returns the conflict indices of an element as a range, such as a
     * std::vector or a boost::container::small_vector. Conflict indices
     * are, for example, the degrees of freedom a cell writes to. All
     * conflict indices must lie in [0, @p n_indices).
     *
     * @note The coloring is greedy and supports 64 colors. A chunk that
     * conflicts with all of them is placed in a color of its own. The
     * result does not depend on the number of threads.
     *
     * @note The conflict indices are a rank-local numbering.
     */
    template <typename Iterator, typename ConflictIndices>
    std::vector<std::vector<dealii::IteratorRange<Iterator>>>
    make_colored_chunks(const Iterator &begin,
                        const dealii::std_cxx20::type_identity_t<Iterator> &end,
                        const unsigned int chunk_size,
                        const std::size_t n_indices,
                        const ConflictIndices &get_conflict_indices)
    {
      Assert(chunk_size > 0, dealii::ExcMessage("chunk_size must be > 0"));

      /*
       * We start by splitting the given range [begin, end) into consecutive
       * chunks of chunk_size elements ...
       */

      std::vector<dealii::IteratorRange<Iterator>> chunks;
      for (Iterator first = begin; first != end;) {
        Iterator last = first;
        for (unsigned int i = 0; i < chunk_size && last != end; ++i)
          ++last;
        chunks.emplace_back(first, last);
        first = last;
      }

      /*
       * ... and collect the conflict indices of every chunk, which we sort
       * and make unique. Chunks are independent, so we do this in parallel:
       */

      std::vector<std::vector<dealii::types::global_dof_index>> indices(
          chunks.size());

      dealii::parallel::apply_to_subranges(
          std::size_t(0),
          chunks.size(),
          [&](const std::size_t first, const std::size_t last) {
            for (std::size_t c = first; c < last; ++c) {
              auto &chunk_indices = indices[c];
              for (const auto &it : chunks[c]) {
                const auto element_indices = get_conflict_indices(it);
                chunk_indices.insert(chunk_indices.end(),
                                     element_indices.begin(),
                                     element_indices.end());
              }
              std::sort(chunk_indices.begin(), chunk_indices.end());
              chunk_indices.erase(
                  std::unique(chunk_indices.begin(), chunk_indices.end()),
                  chunk_indices.end());
            }
          },
          1);

      /*
       * We then color the chunks greedily in a fixed order: For every index
       * we store a bit mask of the colors that write to it, and a chunk
       * takes the lowest color that none of its indices carries. A chunk
       * that conflicts with all 64 colors is placed in a color of its own
       * at the end:
       */

      std::vector<std::uint64_t> colors_of_index(n_indices);
      std::vector<std::vector<dealii::IteratorRange<Iterator>>> colors;
      std::vector<std::vector<dealii::IteratorRange<Iterator>>> overflow;

      for (std::size_t c = 0; c < chunks.size(); ++c) {
        std::uint64_t used_colors = 0;
        for (const auto i : indices[c]) {
          AssertIndexRange(i, n_indices);
          used_colors |= colors_of_index[i];
        }

        const unsigned int color = std::countr_one(used_colors);
        if (color == 64) {
          overflow.push_back({chunks[c]});
          continue;
        }

        for (const auto i : indices[c])
          colors_of_index[i] |= std::uint64_t(1) << color;

        if (color >= colors.size())
          colors.resize(color + 1);
        colors[color].push_back(chunks[c]);
      }

      for (auto &color : overflow)
        colors.push_back(std::move(color));
      return colors;
    }


    /**
     * Variant of the colored dealii::WorkStream::run() that operates on
     * chunks created by make_colored_chunks(). The chunks of a color are
     * processed in parallel, colors one after the other. Within a chunk,
     * @p worker and @p copier are called for every element in order, with
     * a fresh copy of @p sample_scratch_data and @p sample_copy_data for
     * every chunk. The result therefore does not depend on the number of
     * threads.
     *
     * As in the colored dealii::WorkStream::run(), @p copier runs
     * concurrently for the chunks of a color.
     */
    template <typename Iterator,
              typename Worker,
              typename Copier,
              typename ScratchData,
              typename CopyData>
    void run(const std::vector<std::vector<dealii::IteratorRange<Iterator>>>
                 &colored_chunks,
             const Worker &worker,
             const Copier &copier,
             const ScratchData &sample_scratch_data,
             const CopyData &sample_copy_data)
    {
      struct Empty {};

      const auto chunk_worker =
          [&](const dealii::IteratorRange<Iterator> &chunk, Empty &, Empty &) {
            ScratchData scratch_data(sample_scratch_data);
            CopyData copy_data(sample_copy_data);
            for (const auto &it : chunk) {
              worker(it, scratch_data, copy_data);
              copier(copy_data);
            }
          };

      dealii::WorkStream::run(colored_chunks,
                              chunk_worker,
                              std::function<void(const Empty &)>(),
                              Empty(),
                              Empty(),
                              /* queue_length, ignored */ 1,
                              /* chunk_size */ 1);
    }
  } // namespace WorkStream
} // namespace ryujin
