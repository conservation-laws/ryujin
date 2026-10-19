//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/library.h>
#include <ryujin/euler_aeos/equation_of_state.h>

#include <string>
#include <vector>

namespace ryujin
{
  /**
   * The library of equations of state: a registry of all equations of
   * state that the EulerAEOS::HyperbolicSystem can select by name. The
   * built-in equations of state of the namespace EquationsOfState are
   * registered by default.
   *
   * @ingroup EulerAEOSEquations
   */
  using EquationOfStateLibrary = Library<EquationOfState, const std::string &>;

#ifndef DOXYGEN
  template <>
  auto EquationOfStateLibrary::defaults() -> std::vector<Creator>;

  extern template class Library<EquationOfState, const std::string &>;
#endif
} // namespace ryujin
