//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#include <ryujin/scalar_conservation/hyperbolic_description.h>

#include <ryujin/typed/solution_transfer.template.h>

namespace ryujin
{
  using HyperbolicDescription = ScalarConservation::HyperbolicDescription;

  template class SolutionTransfer<HyperbolicDescription, 1, NUMBER>;
  template class SolutionTransfer<HyperbolicDescription, 2, NUMBER>;
  template class SolutionTransfer<HyperbolicDescription, 3, NUMBER>;
} // namespace ryujin
