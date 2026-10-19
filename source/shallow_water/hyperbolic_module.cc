//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#include <ryujin/shallow_water/hyperbolic_description.h>

#include <ryujin/typed/hyperbolic_module.template.h>

namespace ryujin
{
  using HyperbolicDescription = ShallowWater::HyperbolicDescription;

  /* instantiations */

  template class HyperbolicModule<HyperbolicDescription, 1, NUMBER>;
  template class HyperbolicModule<HyperbolicDescription, 2, NUMBER>;
  template class HyperbolicModule<HyperbolicDescription, 3, NUMBER>;
} /* namespace ryujin */
