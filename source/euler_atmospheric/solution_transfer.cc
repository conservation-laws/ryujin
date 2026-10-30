//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// [LANL Copyright Statement]
// Copyright (C) 2024 - 2026 by the ryujin authors
// Copyright (C) 2025 - 2026 by Triad National Security, LLC
//

#include <ryujin/euler_atmospheric/hyperbolic_description.h>
#include <ryujin/typed/solution_transfer.template.h>

namespace ryujin
{
  using HyperbolicDescription = EulerAtmospheric::HyperbolicDescription;

  template class SolutionTransfer<HyperbolicDescription, 1, NUMBER>;
  template class SolutionTransfer<HyperbolicDescription, 2, NUMBER>;
  template class SolutionTransfer<HyperbolicDescription, 3, NUMBER>;
} // namespace ryujin
