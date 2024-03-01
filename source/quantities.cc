//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#include "quantities.template.h"
#include <instantiate.h>

namespace ryujin
{
  /* instantiations */
  template class Quantities<Description, 1, NUMBER>;
  template class Quantities<Description, 2, NUMBER>;
  template class Quantities<Description, 3, NUMBER>;

} /* namespace ryujin */
