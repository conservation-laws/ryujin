//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#include <compile_time_options.h>

#include "problem_description.h"
#include "sparse_matrix_simd.template.h"

namespace ryujin
{
  /* instantiations */

  template class SparsityPatternSIMD<dealii::VectorizedArray<NUMBER>::size()>;

  template class SparseMatrixSIMD<NUMBER>;

#if DIM != 1
  template class SparseMatrixSIMD<NUMBER, DIM>;
#endif

  template class SparseMatrixSIMD<NUMBER,
                                  ProblemDescription::problem_dimension<DIM>>;

} /* namespace ryujin */
