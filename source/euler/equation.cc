//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#include <ryujin/euler/hyperbolic_description.h>
#include <ryujin/stub/parabolic_module.h>
#include <ryujin/typed/equation.template.h>

namespace ryujin
{
  /* instantiations */

  template class Equation<Euler::HyperbolicDescription,
                          StubParabolicDescription,
                          NUMBER>;
} // namespace ryujin
