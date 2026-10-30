//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// [LANL Copyright Statement]
// Copyright (C) 2023 - 2026 by the ryujin authors
// Copyright (C) 2025 - 2026 by Triad National Security, LLC
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/euler_atmospheric/hyperbolic_system.h>
#include <ryujin/typed/initial_state.h>

#include <string>
#include <vector>

namespace ryujin
{
#ifndef DOXYGEN
  template <>
  auto
  InitialStateLibrary<EulerAtmospheric::HyperbolicSystem, 1, NUMBER>::defaults()
      -> std::vector<Creator>;
  template <>
  auto
  InitialStateLibrary<EulerAtmospheric::HyperbolicSystem, 2, NUMBER>::defaults()
      -> std::vector<Creator>;
  template <>
  auto
  InitialStateLibrary<EulerAtmospheric::HyperbolicSystem, 3, NUMBER>::defaults()
      -> std::vector<Creator>;

  extern template class Library<
      InitialState<EulerAtmospheric::HyperbolicSystem, 1, NUMBER>,
      const EulerAtmospheric::HyperbolicSystem &,
      const std::string &>;
  extern template class Library<
      InitialState<EulerAtmospheric::HyperbolicSystem, 2, NUMBER>,
      const EulerAtmospheric::HyperbolicSystem &,
      const std::string &>;
  extern template class Library<
      InitialState<EulerAtmospheric::HyperbolicSystem, 3, NUMBER>,
      const EulerAtmospheric::HyperbolicSystem &,
      const std::string &>;
#endif
} // namespace ryujin
