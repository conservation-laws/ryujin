//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#include "scheduler.template.h"

namespace ryujin
{
  /* instantiations */
  template class Scheduler<1, NUMBER>;
  template class Scheduler<2, NUMBER>;
  template class Scheduler<3, NUMBER>;
} // namespace ryujin
