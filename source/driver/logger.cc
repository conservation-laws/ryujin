//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#include "logger.template.h"

namespace ryujin
{
  /* instantiations */
  template class Logger<1, NUMBER>;
  template class Logger<2, NUMBER>;
  template class Logger<3, NUMBER>;
} // namespace ryujin
