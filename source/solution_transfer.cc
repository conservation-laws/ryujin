//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#include "solution_transfer.template.h"

#include <instantiate.h>

namespace ryujin
{
  template class SolutionTransfer<Description, 1, NUMBER>;
  template class SolutionTransfer<Description, 2, NUMBER>;
  template class SolutionTransfer<Description, 3, NUMBER>;
} // namespace ryujin
