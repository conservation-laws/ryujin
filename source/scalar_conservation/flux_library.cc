//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#include <ryujin/scalar_conservation/flux_burgers.h>
#include <ryujin/scalar_conservation/flux_function.h>
#include <ryujin/scalar_conservation/flux_kpp.h>
#include <ryujin/scalar_conservation/flux_library.h>

namespace ryujin
{
  template <>
  auto FluxLibrary::defaults() -> std::vector<Creator>
  {
    using namespace Fluxes;
    return {
        creator<Burgers>(),
        creator<Function>(),
        creator<KPP>(),
    };
  }

  /* instantiations */

  template class Library<Flux, const std::string &>;
} // namespace ryujin
