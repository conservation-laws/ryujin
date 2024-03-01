//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#include "vtu_output.template.h"
#include <instantiate.h>

namespace ryujin
{
  /* instantiations */
  template class VTUOutput<Description, 1, NUMBER>;
  template class VTUOutput<Description, 2, NUMBER>;
  template class VTUOutput<Description, 3, NUMBER>;

} /* namespace ryujin */
