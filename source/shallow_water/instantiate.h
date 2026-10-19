//
// SPDX-License-Identifier: Apache-2.0
// [LANL Copyright Statement]
// Copyright (C) 2023 - 2024 by the ryujin authors
// Copyright (C) 2023 - 2024 by Triad National Security, LLC
//

#ifndef RYUJIN_INCLUDE_INSTANTIATION_ONCE
#define RYUJIN_INCLUDE_INSTANTIATION_ONCE
#else
#error Instantiation files can only be included once.
#endif

#include <ryujin/shallow_water/hyperbolic_description.h>

#include <ryujin/stub/parabolic_module.h>

namespace ryujin
{
  using HyperbolicDescription = ShallowWater::HyperbolicDescription;
  using ParabolicDescription = StubParabolicDescription;
} // namespace ryujin
