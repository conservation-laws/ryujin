//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#include <ryujin/driver/equation.h>
#include <ryujin/euler_barotropic/hyperbolic_description.h>

namespace ryujin
{
  const EquationRegistration<EulerBarotropic::HyperbolicDescription>
      registration("euler barotropic");
} // namespace ryujin
