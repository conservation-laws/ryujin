//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#ifndef RYUJIN_INCLUDE_INSTANTIATION_ONCE
#define RYUJIN_INCLUDE_INSTANTIATION_ONCE
#else
#error Instantiation files can only be included once.
#endif

#include <ryujin/euler_aeos/hyperbolic_description.h>
#include <ryujin/euler_poisson/parabolic_description.h>

namespace ryujin
{
  using HyperbolicDescription = EulerAEOS::HyperbolicDescription;
  using ParabolicDescription = EulerPoisson::ParabolicDescription;
} // namespace ryujin
