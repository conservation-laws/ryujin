//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#include <ryujin/driver/equation.h>
#include <ryujin/shallow_water/hyperbolic_description.h>

namespace ryujin
{
  const EquationRegistration<ShallowWater::HyperbolicDescription>
      registration("shallow water");
} // namespace ryujin
