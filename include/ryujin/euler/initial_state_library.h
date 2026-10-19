//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/euler/hyperbolic_description.h>
#include <ryujin/typed/initial_state.h>

#include <string>
#include <vector>

namespace ryujin
{
#ifndef DOXYGEN
  template <>
  auto InitialStateLibrary<Euler::HyperbolicDescription, 1, NUMBER>::defaults()
      -> std::vector<Creator>;
  template <>
  auto InitialStateLibrary<Euler::HyperbolicDescription, 2, NUMBER>::defaults()
      -> std::vector<Creator>;
  template <>
  auto InitialStateLibrary<Euler::HyperbolicDescription, 3, NUMBER>::defaults()
      -> std::vector<Creator>;

  extern template class Library<
      InitialState<Euler::HyperbolicDescription, 1, NUMBER>,
      const Euler::HyperbolicSystem &,
      const std::string &>;
  extern template class Library<
      InitialState<Euler::HyperbolicDescription, 2, NUMBER>,
      const Euler::HyperbolicSystem &,
      const std::string &>;
  extern template class Library<
      InitialState<Euler::HyperbolicDescription, 3, NUMBER>,
      const Euler::HyperbolicSystem &,
      const std::string &>;
#endif
} // namespace ryujin
