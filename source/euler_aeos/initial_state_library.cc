//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 by the ryujin authors
//

#include <ryujin/euler_aeos/initial_state_library.template.h>

namespace ryujin
{
  template class InitialStateLibrary<HyperbolicDescription, 1, NUMBER>;
  template class InitialStateLibrary<HyperbolicDescription, 2, NUMBER>;
  template class InitialStateLibrary<HyperbolicDescription, 3, NUMBER>;
} // namespace ryujin
