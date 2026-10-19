//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#include <ryujin/euler_barotropic/hyperbolic_description.h>
#include <ryujin/stub/parabolic_module.h>
#include <ryujin/typed/equation.template.h>

namespace ryujin
{
  /* instantiations */

  template class Equation<EulerBarotropic::HyperbolicDescription,
                          StubParabolicDescription,
                          NUMBER>;
} // namespace ryujin
