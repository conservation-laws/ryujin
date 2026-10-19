//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/library.h>
#include <ryujin/interface/equation.h>

#include <vector>

namespace ryujin
{
  /**
   * The library of equations: a registry of all pairings of a hyperbolic
   * and a parabolic description that the EquationDispatch can select by
   * name. The equations of ryujin are registered by default.
   *
   * A client can add an equation of its own with
   * @code
   * EquationLibrary::add([]() {
   *   return std::make_unique<
   *       Equation<HyperbolicDescription, ParabolicDescription, Number>>(
   *       "equation");
   * });
   * @endcode
   *
   * @ingroup Driver
   */
  using EquationLibrary = Library<Interface::Equation>;

#ifndef DOXYGEN
  template <>
  auto EquationLibrary::defaults() -> std::vector<Creator>;

  extern template class Library<Interface::Equation>;
#endif
} // namespace ryujin
