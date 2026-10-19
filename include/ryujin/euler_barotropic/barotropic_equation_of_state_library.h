//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/library.h>
#include <ryujin/euler_barotropic/barotropic_equation_of_state.h>

#include <string>
#include <vector>

namespace ryujin
{
  /**
   * The library of barotropic equations of state: a registry of all
   * barotropic equations of state that the
   * EulerBarotropic::HyperbolicSystem can select by name. The built-in
   * equations of state of the namespace BarotropicEquationsOfState are
   * registered by default.
   *
   * @ingroup EulerBarotropicEquations
   */
  using BarotropicEquationOfStateLibrary =
      Library<BarotropicEquationOfState, const std::string &>;

#ifndef DOXYGEN
  template <>
  auto BarotropicEquationOfStateLibrary::defaults() -> std::vector<Creator>;

  extern template class Library<BarotropicEquationOfState, const std::string &>;
#endif
} // namespace ryujin
