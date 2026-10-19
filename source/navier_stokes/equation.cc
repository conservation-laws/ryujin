//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#include <ryujin/euler/hyperbolic_description.h>
#include <ryujin/navier_stokes/parabolic_description.h>
#include <ryujin/typed/equation.template.h>

namespace ryujin
{
  /* instantiations */

  template class Equation<Euler::HyperbolicDescription,
                          NavierStokes::ParabolicDescription,
                          NUMBER>;
} // namespace ryujin
