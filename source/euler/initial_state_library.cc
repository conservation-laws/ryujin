//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#include "initial_state_library.template.h"

namespace ryujin
{
  namespace Euler
  {
    /* instantiations */
    template void InitialStateLibrary::populate_initial_state_list(
        initial_state_list_type<DIM, NUMBER> &,
        const HyperbolicSystem::View<DIM, NUMBER> &,
        const std::string &);

  } // namespace Euler
} // namespace ryujin
