//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/euler_barotropic/hyperbolic_description.h>
#include <ryujin/typed/initial_state.h>

#include <string>
#include <vector>

namespace ryujin
{
#ifndef DOXYGEN
  template <>
  auto InitialStateLibrary<EulerBarotropic::HyperbolicDescription, 1, NUMBER>::
      defaults() -> std::vector<Creator>;
  template <>
  auto InitialStateLibrary<EulerBarotropic::HyperbolicDescription, 2, NUMBER>::
      defaults() -> std::vector<Creator>;
  template <>
  auto InitialStateLibrary<EulerBarotropic::HyperbolicDescription, 3, NUMBER>::
      defaults() -> std::vector<Creator>;

  extern template class Library<
      InitialState<EulerBarotropic::HyperbolicDescription, 1, NUMBER>,
      const EulerBarotropic::HyperbolicSystem &,
      const std::string &>;
  extern template class Library<
      InitialState<EulerBarotropic::HyperbolicDescription, 2, NUMBER>,
      const EulerBarotropic::HyperbolicSystem &,
      const std::string &>;
  extern template class Library<
      InitialState<EulerBarotropic::HyperbolicDescription, 3, NUMBER>,
      const EulerBarotropic::HyperbolicSystem &,
      const std::string &>;
#endif
} // namespace ryujin
