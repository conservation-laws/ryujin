//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2025 by the ryujin authors
//

#ifndef RYUJIN_INCLUDE_INSTANTIATION_ONCE
#define RYUJIN_INCLUDE_INSTANTIATION_ONCE
#else
#error Instantiation files can only be included once.
#endif

#include <ryujin/euler_barotropic/hyperbolic_description.h>

#include <ryujin/stub/parabolic_module.h>

namespace ryujin
{
  using HyperbolicDescription = EulerBarotropic::HyperbolicDescription;
  using ParabolicDescription = StubParabolicDescription;
} // namespace ryujin
