//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#include <ryujin/driver/equation.h>
#include <ryujin/euler/hyperbolic_description.h>
#include <ryujin/euler_poisson/parabolic_description.h>

namespace ryujin
{
  const EquationRegistration<Euler::HyperbolicDescription,
                             EulerPoisson::ParabolicDescription>
      registration("euler poisson");
} // namespace ryujin
