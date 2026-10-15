//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include "error_evaluation.template.h"
#include <instantiate.h>

namespace ryujin
{
  /* instantiations */
  template class ErrorEvaluation<HyperbolicDescription, 1, NUMBER>;
  template class ErrorEvaluation<HyperbolicDescription, 2, NUMBER>;
  template class ErrorEvaluation<HyperbolicDescription, 3, NUMBER>;

} /* namespace ryujin */
