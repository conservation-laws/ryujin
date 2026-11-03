//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#include "vtu_output.template.h"

namespace ryujin
{
  /* instantiations */
  template class VTUOutput<1, 1, NUMBER>;
  template class VTUOutput<2, 2, NUMBER>;
  template class VTUOutput<3, 3, NUMBER>;
  template class VTUOutput<1, 2, NUMBER>;
  template class VTUOutput<2, 3, NUMBER>;

} // namespace ryujin
