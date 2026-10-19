//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#include <ryujin/euler/hyperbolic_description.h>
#include <ryujin/euler_poisson/parabolic_description.h>

#include <driver/equation_dispatch.h>
#include <ryujin/base/compile_time_options.h>

namespace ryujin
{
  namespace EulerPoisson
  {
    Dispatch<Euler::HyperbolicDescription, ParabolicDescription, NUMBER>
        dispatch_instance("euler poisson");
  } // namespace EulerPoisson
} // namespace ryujin
