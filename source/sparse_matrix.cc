//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#include <compile_time_options.h>

#include "sparse_matrix.template.h"

namespace ryujin
{
  /* instantiations */

  template class SparseMatrix<NUMBER, 1>;
  template class SparseMatrix<NUMBER, 2>;
  template class SparseMatrix<NUMBER, 3>;
} /* namespace ryujin */
