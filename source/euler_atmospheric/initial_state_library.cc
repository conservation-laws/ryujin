//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// [LANL Copyright Statement]
// Copyright (C) 2023 - 2026 by the ryujin authors
// Copyright (C) 2025 - 2026 by Triad National Security, LLC
//

#include <euler_atmospheric/initial_state_defaults.h>
#include <ryujin/euler_atmospheric/initial_state_library.h>

namespace ryujin
{
  template <>
  auto
  InitialStateLibrary<EulerAtmospheric::HyperbolicSystem, 1, NUMBER>::defaults()
      -> std::vector<Creator>
  {
    return EulerAtmosphericInitialStates::
        defaults<EulerAtmospheric::HyperbolicSystem, 1, NUMBER>();
  }

  template <>
  auto
  InitialStateLibrary<EulerAtmospheric::HyperbolicSystem, 2, NUMBER>::defaults()
      -> std::vector<Creator>
  {
    return EulerAtmosphericInitialStates::
        defaults<EulerAtmospheric::HyperbolicSystem, 2, NUMBER>();
  }

  template <>
  auto
  InitialStateLibrary<EulerAtmospheric::HyperbolicSystem, 3, NUMBER>::defaults()
      -> std::vector<Creator>
  {
    return EulerAtmosphericInitialStates::
        defaults<EulerAtmospheric::HyperbolicSystem, 3, NUMBER>();
  }

  /* instantiations */

  template class Library<
      InitialState<EulerAtmospheric::HyperbolicSystem, 1, NUMBER>,
      const EulerAtmospheric::HyperbolicSystem &,
      const std::string &>;
  template class Library<
      InitialState<EulerAtmospheric::HyperbolicSystem, 2, NUMBER>,
      const EulerAtmospheric::HyperbolicSystem &,
      const std::string &>;
  template class Library<
      InitialState<EulerAtmospheric::HyperbolicSystem, 3, NUMBER>,
      const EulerAtmospheric::HyperbolicSystem &,
      const std::string &>;
} // namespace ryujin
