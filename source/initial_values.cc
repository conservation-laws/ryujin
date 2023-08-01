//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#include "initial_values.template.h"
#include <instantiate.h>

namespace ryujin
{
  /* instantiations */
  template class InitialValues<Description, 1, NUMBER>;
  template class InitialValues<Description, 2, NUMBER>;
  template class InitialValues<Description, 3, NUMBER>;

} /* namespace ryujin */
