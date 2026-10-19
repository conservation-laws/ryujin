//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#include <ryujin/scalar_conservation/hyperbolic_description.h>
#include <ryujin/scalar_conservation/initial_state_library.h>
#include <ryujin/typed/initial_values.template.h>

namespace ryujin
{
  using HyperbolicDescription = ScalarConservation::HyperbolicDescription;

  /* instantiations */
  template class InitialValues<HyperbolicDescription, 1, NUMBER>;
  template class InitialValues<HyperbolicDescription, 2, NUMBER>;
  template class InitialValues<HyperbolicDescription, 3, NUMBER>;

} // namespace ryujin
