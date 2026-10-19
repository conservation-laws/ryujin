//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#include <ryujin/euler_barotropic/hyperbolic_description.h>
#include <ryujin/euler_poisson/parabolic_description.h>

#include <equation_dispatch.h>
#include <ryujin/base/compile_time_options.h>

namespace ryujin
{
  namespace EulerPoissonBarotropic
  {
    Dispatch<EulerBarotropic::HyperbolicDescription,
             EulerPoisson::ParabolicDescription,
             NUMBER>
        dispatch_instance("euler poisson barotropic");
  } // namespace EulerPoissonBarotropic
} // namespace ryujin
