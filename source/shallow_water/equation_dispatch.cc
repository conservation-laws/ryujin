//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#include <ryujin/shallow_water/hyperbolic_description.h>

#include <ryujin/stub/parabolic_module.h>

#include <driver/equation_dispatch.h>
#include <ryujin/base/compile_time_options.h>

namespace ryujin
{
  namespace ShallowWater
  {
    Dispatch<HyperbolicDescription, StubParabolicDescription, NUMBER>
        dispatch_instance("shallow water");
  } // namespace ShallowWater
} // namespace ryujin
