//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#include <euler/initial_state_defaults.h>
#include <ryujin/euler_aeos/initial_state_library.h>

namespace ryujin
{
  template <>
  auto
  InitialStateLibrary<EulerAEOS::HyperbolicDescription, 1, NUMBER>::defaults()
      -> std::vector<Creator>
  {
    return EulerInitialStates::
        defaults<EulerAEOS::HyperbolicDescription, 1, NUMBER>();
  }

  template <>
  auto
  InitialStateLibrary<EulerAEOS::HyperbolicDescription, 2, NUMBER>::defaults()
      -> std::vector<Creator>
  {
    return EulerInitialStates::
        defaults<EulerAEOS::HyperbolicDescription, 2, NUMBER>();
  }

  template <>
  auto
  InitialStateLibrary<EulerAEOS::HyperbolicDescription, 3, NUMBER>::defaults()
      -> std::vector<Creator>
  {
    return EulerInitialStates::
        defaults<EulerAEOS::HyperbolicDescription, 3, NUMBER>();
  }

  /* instantiations */

  template class Library<
      InitialState<EulerAEOS::HyperbolicDescription, 1, NUMBER>,
      const EulerAEOS::HyperbolicSystem &,
      const std::string &>;
  template class Library<
      InitialState<EulerAEOS::HyperbolicDescription, 2, NUMBER>,
      const EulerAEOS::HyperbolicSystem &,
      const std::string &>;
  template class Library<
      InitialState<EulerAEOS::HyperbolicDescription, 3, NUMBER>,
      const EulerAEOS::HyperbolicSystem &,
      const std::string &>;
} // namespace ryujin
