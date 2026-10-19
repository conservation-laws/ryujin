//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/library.h>
#include <ryujin/scalar_conservation/flux.h>

#include <string>
#include <vector>

namespace ryujin
{
  /**
   * The library of fluxes: a registry of all fluxes that the
   * ScalarConservation::HyperbolicSystem can select by name. The built-in
   * fluxes of the namespace Fluxes are registered by default.
   *
   * @ingroup ScalarConservationEquations
   */
  using FluxLibrary = Library<Flux, const std::string &>;

#ifndef DOXYGEN
  template <>
  auto FluxLibrary::defaults() -> std::vector<Creator>;

  extern template class Library<Flux, const std::string &>;
#endif
} // namespace ryujin
