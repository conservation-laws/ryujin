//
// SPDX-License-Identifier: Apache-2.0
// [LANL Copyright Statement]
// Copyright (C) 2023 - 2024 by the ryujin authors
// Copyright (C) 2023 - 2024 by Triad National Security, LLC
//

#include "initial_state_library.template.h"

namespace ryujin
{
  template class InitialStateLibrary<HyperbolicDescription, 1, NUMBER>;
  template class InitialStateLibrary<HyperbolicDescription, 2, NUMBER>;
  template class InitialStateLibrary<HyperbolicDescription, 3, NUMBER>;
} // namespace ryujin
