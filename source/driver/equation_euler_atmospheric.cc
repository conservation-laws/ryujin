//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// [LANL Copyright Statement]
// Copyright (C) 2024 - 2026 by the ryujin authors
// Copyright (C) 2025 by Triad National Security, LLC
//

#include <ryujin/driver/equation.h>
#include <ryujin/euler_atmospheric/hyperbolic_description.h>

namespace ryujin
{
  const EquationRegistration<EulerAtmospheric::HyperbolicDescription>
      registration("euler atmospheric");
} // namespace ryujin
