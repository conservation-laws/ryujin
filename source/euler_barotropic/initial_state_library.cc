//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#include <euler/initial_state_defaults.h>
#include <ryujin/euler_barotropic/initial_state_library.h>

namespace ryujin
{
  template <>
  auto
  InitialStateLibrary<EulerBarotropic::HyperbolicSystem, 1, NUMBER>::defaults()
      -> std::vector<Creator>
  {
    return EulerInitialStates::
        defaults<EulerBarotropic::HyperbolicSystem, 1, NUMBER>();
  }

  template <>
  auto
  InitialStateLibrary<EulerBarotropic::HyperbolicSystem, 2, NUMBER>::defaults()
      -> std::vector<Creator>
  {
    return EulerInitialStates::
        defaults<EulerBarotropic::HyperbolicSystem, 2, NUMBER>();
  }

  template <>
  auto
  InitialStateLibrary<EulerBarotropic::HyperbolicSystem, 3, NUMBER>::defaults()
      -> std::vector<Creator>
  {
    return EulerInitialStates::
        defaults<EulerBarotropic::HyperbolicSystem, 3, NUMBER>();
  }

  /* instantiations */

  template class Library<
      InitialState<EulerBarotropic::HyperbolicSystem, 1, NUMBER>,
      const EulerBarotropic::HyperbolicSystem &,
      const std::string &>;
  template class Library<
      InitialState<EulerBarotropic::HyperbolicSystem, 2, NUMBER>,
      const EulerBarotropic::HyperbolicSystem &,
      const std::string &>;
  template class Library<
      InitialState<EulerBarotropic::HyperbolicSystem, 3, NUMBER>,
      const EulerBarotropic::HyperbolicSystem &,
      const std::string &>;
} // namespace ryujin
