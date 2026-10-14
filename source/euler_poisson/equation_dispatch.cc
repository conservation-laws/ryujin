//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#include "../euler/hyperbolic_description.h"
#include "parabolic_description.h"

#include <compile_time_options.h>
#include <equation_dispatch.h>

namespace ryujin
{
  namespace EulerPoisson
  {
    Dispatch<Euler::HyperbolicDescription, ParabolicDescription, NUMBER>
        dispatch_instance("euler poisson");
  } // namespace EulerPoisson
} // namespace ryujin
