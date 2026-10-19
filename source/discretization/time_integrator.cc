//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#include "time_integrator.template.h"

namespace ryujin
{
  /* instantiations */
  template class TimeIntegrator<1, NUMBER>;
  template class TimeIntegrator<2, NUMBER>;
  template class TimeIntegrator<3, NUMBER>;
} // namespace ryujin
