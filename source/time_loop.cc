//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#include "time_loop.template.h"
#include <instantiate.h>

namespace ryujin
{
  /* instantiations */
  template class TimeLoop<Description, 1, NUMBER>;
  template class TimeLoop<Description, 2, NUMBER>;
  template class TimeLoop<Description, 3, NUMBER>;

} // namespace ryujin
