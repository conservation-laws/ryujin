//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#include <ryujin/euler_barotropic/hyperbolic_description.h>

#include <ryujin/typed/solution_transfer.template.h>

namespace ryujin
{
  using HyperbolicDescription = EulerBarotropic::HyperbolicDescription;

  template class SolutionTransfer<HyperbolicDescription, 1, NUMBER>;
  template class SolutionTransfer<HyperbolicDescription, 2, NUMBER>;
  template class SolutionTransfer<HyperbolicDescription, 3, NUMBER>;
} // namespace ryujin
