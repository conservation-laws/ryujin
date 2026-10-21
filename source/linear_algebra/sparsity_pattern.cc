//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#include <ryujin/base/compile_time_options.h>

#include "sparsity_pattern.template.h"

#include <deal.II/base/vectorization.h>

namespace ryujin
{
  /* instantiations */

  template class SparsityPattern<warp_size>;

#if DEAL_II_VECTORIZATION_WIDTH_IN_BITS > 0
  using VA = dealii::VectorizedArray<double>;
  constexpr auto simd_width = VA::size();
  template class SparsityPattern<simd_width == warp_size ? 1 : simd_width>;
#endif
} // namespace ryujin
