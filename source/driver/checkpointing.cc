//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include "checkpointing.template.h"

namespace ryujin
{
  /* instantiations */
  template class Checkpointing<1, NUMBER>;
  template class Checkpointing<2, NUMBER>;
  template class Checkpointing<3, NUMBER>;
} // namespace ryujin
