//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 by the ryujin authors
//

#include <ryujin/euler/hyperbolic_description.h>
#include <ryujin/navier_stokes/parabolic_description.h>
#include <ryujin/navier_stokes/parabolic_module.template.h>

namespace ryujin
{
  namespace NavierStokes
  {
    template class ParabolicModule<1, NUMBER>;
    template class ParabolicModule<2, NUMBER>;
    template class ParabolicModule<3, NUMBER>;
  } // namespace NavierStokes
} // namespace ryujin
