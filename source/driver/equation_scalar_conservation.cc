//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#include <ryujin/driver/equation.h>
#include <ryujin/scalar_conservation/hyperbolic_description.h>

namespace ryujin
{
  const EquationRegistration<ScalarConservation::HyperbolicDescription>
      registration("scalar conservation");
} // namespace ryujin
