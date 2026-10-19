//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#include <ryujin/base/compile_time_options.h>
#include <ryujin/euler/hyperbolic_description.h>
#include <ryujin/navier_stokes/parabolic_description.h>

#include <driver/equation_dispatch.h>

namespace ryujin
{
  namespace NavierStokes
  {
    Dispatch<Euler::HyperbolicDescription, ParabolicDescription, NUMBER>
        dispatch_instance("navier stokes");
  } // namespace NavierStokes
} // namespace ryujin
