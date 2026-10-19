//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#include "time_loop.template.h"

namespace ryujin
{
  /* instantiations */
  template class TimeLoop<1, NUMBER>;
  template class TimeLoop<2, NUMBER>;
  template class TimeLoop<3, NUMBER>;
} // namespace ryujin
