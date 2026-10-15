//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#include "../euler_barotropic/hyperbolic_description.h"
#include "../euler_poisson/parabolic_description.h"

#include <compile_time_options.h>
#include <equation_dispatch.h>

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
