//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/scalar_diffusion/parabolic_description.h>
#include <ryujin/scalar_diffusion/parabolic_module.template.h>

namespace ryujin
{
  namespace ScalarDiffusion
  {
    template class ParabolicModule<1, NUMBER>;
    template class ParabolicModule<2, NUMBER>;
    template class ParabolicModule<3, NUMBER>;
  } // namespace ScalarDiffusion
} // namespace ryujin
