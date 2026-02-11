//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#include <compile_time_options.h>

#include "sparsity_pattern.template.h"

namespace ryujin
{
  /* instantiations */

  template class SparsityPattern<dealii::VectorizedArray<NUMBER>::size()>;
} /* namespace ryujin */
