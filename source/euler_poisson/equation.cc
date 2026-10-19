//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#include <ryujin/euler/hyperbolic_description.h>
#include <ryujin/euler_poisson/parabolic_description.h>
#include <ryujin/typed/equation.template.h>

namespace ryujin
{
  /* instantiations */

  template class Equation<Euler::HyperbolicDescription,
                          EulerPoisson::ParabolicDescription,
                          NUMBER>;
} // namespace ryujin
