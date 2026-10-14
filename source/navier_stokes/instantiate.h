//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 by the ryujin authors
//

#ifndef RYUJIN_INCLUDE_INSTANTIATION_ONCE
#define RYUJIN_INCLUDE_INSTANTIATION_ONCE
#else
#error Instantiation files can only be included once.
#endif

#include "../euler/hyperbolic_description.h"
#include "parabolic_description.h"

namespace ryujin
{
  using HyperbolicDescription = Euler::HyperbolicDescription;
  using ParabolicDescription = NavierStokes::ParabolicDescription;
} // namespace ryujin
