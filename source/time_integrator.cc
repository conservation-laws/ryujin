//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#include "time_integrator.template.h"
#include <instantiate.h>

namespace ryujin
{
  /* instantiations */
  using HD = HyperbolicDescription;
  using PD = ParabolicDescription;

  template class TimeIntegrator<HD, PD, 1, NUMBER>;
  template class TimeIntegrator<HD, PD, 2, NUMBER>;
  template class TimeIntegrator<HD, PD, 3, NUMBER>;

} /* namespace ryujin */
