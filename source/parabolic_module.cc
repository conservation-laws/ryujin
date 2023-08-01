//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#include "parabolic_module.template.h"
#include <instantiate.h>

namespace ryujin
{
  /* instantiations */
  template class ParabolicModule<Description, 1, NUMBER>;
  template class ParabolicModule<Description, 2, NUMBER>;
  template class ParabolicModule<Description, 3, NUMBER>;

} /* namespace ryujin */
