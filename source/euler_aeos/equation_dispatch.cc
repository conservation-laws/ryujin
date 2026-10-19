//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#include <ryujin/euler_aeos/hyperbolic_description.h>

#include <ryujin/stub/parabolic_module.h>

#include <driver/equation_dispatch.h>
#include <ryujin/base/compile_time_options.h>

namespace ryujin
{
  namespace EulerAEOS
  {
    Dispatch<HyperbolicDescription, StubParabolicDescription, NUMBER>
        dispatch_instance("euler aeos");
  } // namespace EulerAEOS
} // namespace ryujin
