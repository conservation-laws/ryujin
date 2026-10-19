//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#include <ryujin/euler/hyperbolic_description.h>

#include <ryujin/stub/parabolic_module.h>

#include <equation_dispatch.h>
#include <ryujin/base/compile_time_options.h>

namespace ryujin
{
  namespace Euler
  {
    Dispatch<HyperbolicDescription, StubParabolicDescription, NUMBER>
        dispatch_instance("euler");
  } // namespace Euler
} // namespace ryujin
