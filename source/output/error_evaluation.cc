//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include "error_evaluation.template.h"

namespace ryujin
{
  /* instantiations */
  template class ErrorEvaluation<1, NUMBER>;
  template class ErrorEvaluation<2, NUMBER>;
  template class ErrorEvaluation<3, NUMBER>;

} // namespace ryujin
