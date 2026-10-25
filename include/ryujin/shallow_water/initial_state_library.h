//
// SPDX-License-Identifier: Apache-2.0
// [LANL Copyright Statement]
// Copyright (C) 2023 - 2026 by the ryujin authors
// Copyright (C) 2023 - 2024 by Triad National Security, LLC
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/shallow_water/hyperbolic_system.h>
#include <ryujin/typed/initial_state.h>

#include <string>
#include <vector>

namespace ryujin
{
#ifndef DOXYGEN
  template <>
  auto
  InitialStateLibrary<ShallowWater::HyperbolicSystem, 1, NUMBER>::defaults()
      -> std::vector<Creator>;
  template <>
  auto
  InitialStateLibrary<ShallowWater::HyperbolicSystem, 2, NUMBER>::defaults()
      -> std::vector<Creator>;
  template <>
  auto
  InitialStateLibrary<ShallowWater::HyperbolicSystem, 3, NUMBER>::defaults()
      -> std::vector<Creator>;

  extern template class Library<
      InitialState<ShallowWater::HyperbolicSystem, 1, NUMBER>,
      const ShallowWater::HyperbolicSystem &,
      const std::string &>;
  extern template class Library<
      InitialState<ShallowWater::HyperbolicSystem, 2, NUMBER>,
      const ShallowWater::HyperbolicSystem &,
      const std::string &>;
  extern template class Library<
      InitialState<ShallowWater::HyperbolicSystem, 3, NUMBER>,
      const ShallowWater::HyperbolicSystem &,
      const std::string &>;
#endif
} // namespace ryujin
