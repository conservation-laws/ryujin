//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#include "hyperbolic_module.template.h"
#include <instantiate.h>

namespace ryujin
{
  /* instantiations */
  template class HyperbolicModule<Description, 1, NUMBER>;
  template class HyperbolicModule<Description, 2, NUMBER>;
  template class HyperbolicModule<Description, 3, NUMBER>;

} /* namespace ryujin */
