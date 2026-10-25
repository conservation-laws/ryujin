//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#include <ryujin/euler/hyperbolic_system.h>
#include <ryujin/euler/initial_state_library.h>
#include <ryujin/typed/initial_values.template.h>

namespace ryujin
{
  using HyperbolicSystem = Euler::HyperbolicSystem;

  /* instantiations */
  template class InitialValues<HyperbolicSystem, 1, NUMBER>;
  template class InitialValues<HyperbolicSystem, 2, NUMBER>;
  template class InitialValues<HyperbolicSystem, 3, NUMBER>;

} // namespace ryujin
