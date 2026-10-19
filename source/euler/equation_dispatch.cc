//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#include <driver/equation_dispatch.h>
#include <ryujin/base/compile_time_options.h>
#include <ryujin/euler/hyperbolic_description.h>
#include <ryujin/stub/parabolic_module.h>

namespace ryujin
{
  namespace Euler
  {
    Dispatch<HyperbolicDescription, StubParabolicDescription, NUMBER>
        dispatch_instance("euler");
  } // namespace Euler
} // namespace ryujin
