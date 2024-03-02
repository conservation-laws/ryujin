//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#include <compile_time_options.h>

#include "sparse_matrix_simd.template.h"

namespace ryujin
{
  /* instantiations */

  template class SparsityPatternSIMD<dealii::VectorizedArray<NUMBER>::size()>;

  template class SparseMatrixSIMD<NUMBER, 1>;
  template class SparseMatrixSIMD<NUMBER, 2>;
  template class SparseMatrixSIMD<NUMBER, 3>;
} /* namespace ryujin */
