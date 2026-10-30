//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// [LANL Copyright Statement]
// Copyright (C) 2020 - 2026 by the ryujin authors
// Copyright (C) 2025 - 2026 by Triad National Security, LLC
//

#include <ryujin/euler_atmospheric/hyperbolic_system.h>
#include <ryujin/euler_atmospheric/initial_state_library.h>
#include <ryujin/typed/initial_values.template.h>

namespace ryujin
{
  using HyperbolicSystem = EulerAtmospheric::HyperbolicSystem;

  /* instantiations */
  template class InitialValues<HyperbolicSystem, 1, NUMBER>;
  template class InitialValues<HyperbolicSystem, 2, NUMBER>;
  template class InitialValues<HyperbolicSystem, 3, NUMBER>;

} // namespace ryujin
