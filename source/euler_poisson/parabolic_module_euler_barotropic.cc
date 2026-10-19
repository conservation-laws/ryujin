//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 - 2026 by the ryujin authors
//

#include <ryujin/euler_barotropic/hyperbolic_description.h>
#include <ryujin/euler_poisson/parabolic_module.template.h>

namespace ryujin
{
  namespace EulerPoisson
  {
    using HD = EulerBarotropic::HyperbolicDescription;

    template class ParabolicModule<HD, 1, NUMBER>;
    template class ParabolicModule<HD, 2, NUMBER>;
    template class ParabolicModule<HD, 3, NUMBER>;
  } // namespace EulerPoisson
} // namespace ryujin
