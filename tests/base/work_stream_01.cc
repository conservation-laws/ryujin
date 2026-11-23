//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/base/work_stream.h>

#include <iostream>
#include <numeric>

//
// Test WorkStream::make_colored_chunks() and WorkStream::run(): Element i
// of a chain writes to the entries i and i + 1 of a vector. Check that no
// two chunks of a color share an entry, and that every element is visited
// exactly once. A second coloring where all chunks share index 0 checks
// the overflow beyond 64 colors.
//

using namespace ryujin;

void test(const unsigned int n_elements,
          const unsigned int chunk_size,
          const bool share_index_zero)
{
  std::vector<unsigned int> elements(n_elements);
  std::iota(elements.begin(), elements.end(), 0u);
  using Iterator = std::vector<unsigned int>::const_iterator;

  const auto get_conflict_indices = [&](const Iterator &it) {
    const auto i = *it;
    std::vector<dealii::types::global_dof_index> indices{i + 1};
    indices.push_back(share_index_zero ? 0 : i);
    return indices;
  };

  const auto colors = WorkStream::make_colored_chunks(elements.cbegin(),
                                                      elements.cend(),
                                                      chunk_size,
                                                      n_elements + 1,
                                                      get_conflict_indices);

  std::cout << "n_elements = " << n_elements << ", chunk_size = " << chunk_size
            << ", n_colors = " << colors.size() << std::endl;

  for (const auto &color : colors) {
    /* The chunk of the color that last wrote to an index: */
    std::vector<int> owner(n_elements + 1, -1);
    for (unsigned int c = 0; c < color.size(); ++c)
      for (const auto &it : color[c])
        for (const auto k : get_conflict_indices(it)) {
          if (owner[k] != -1 && owner[k] != int(c))
            std::cout << "conflict at index " << k << std::endl;
          owner[k] = c;
        }
  }

  std::vector<unsigned int> visits(n_elements);
  std::vector<unsigned int> vector(n_elements + 1);

  WorkStream::run(
      colors,
      [&](const Iterator &it, int &, unsigned int &copy) {
        visits[*it]++;
        copy = *it;
      },
      [&](const unsigned int &copy) {
        vector[copy] += 1;
        vector[copy + 1] += 1;
      },
      int(),
      0u);

  for (unsigned int i = 0; i < n_elements; ++i)
    if (visits[i] != 1)
      std::cout << "element " << i << " visited " << visits[i] << " times"
                << std::endl;

  unsigned int sum = 0;
  for (const auto v : vector)
    sum += v;
  std::cout << "sum = " << sum << std::endl;
}


int main()
{
  test(1000, 7, false);
  test(1000, 1, false);
  test(700, 10, true);
}
