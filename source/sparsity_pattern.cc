//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#include <compile_time_options.h>

#include "sparsity_pattern.template.h"

#include <deal.II/base/vectorization.h>

namespace ryujin
{
  /* instantiations */

  template class SparsityPattern<warp_size>;

  /*
   * The testsuite also uses the sparsity pattern with the SIMD width of
   * double precision numbers. Only instantiate if this is a different
   * specialization than the one for the warp size above:
   */
  using VA = dealii::VectorizedArray<double>;
  constexpr auto simd_width = VA::size();
  template class SparsityPattern<simd_width == warp_size ? 1 : simd_width>;
} /* namespace ryujin */
