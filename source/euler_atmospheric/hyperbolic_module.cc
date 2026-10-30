//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// [LANL Copyright Statement]
// Copyright (C) 2020 - 2026 by the ryujin authors
// Copyright (C) 2025 - 2026 by Triad National Security, LLC
//

#include <ryujin/euler_atmospheric/hyperbolic_description.h>
#include <ryujin/typed/hyperbolic_module.template.h>

namespace ryujin
{
  using HyperbolicDescription = EulerAtmospheric::HyperbolicDescription;

  /* instantiations */

  template class HyperbolicModule<HyperbolicDescription, 1, NUMBER>;
  template class HyperbolicModule<HyperbolicDescription, 2, NUMBER>;
  template class HyperbolicModule<HyperbolicDescription, 3, NUMBER>;
} // namespace ryujin
