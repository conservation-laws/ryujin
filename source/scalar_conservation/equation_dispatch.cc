//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#include <ryujin/base/compile_time_options.h>
#include <ryujin/scalar_conservation/hyperbolic_description.h>
#include <ryujin/stub/parabolic_module.h>

#include <driver/equation_dispatch.h>

namespace ryujin
{
  namespace ScalarConservation
  {
    Dispatch<HyperbolicDescription, StubParabolicDescription, NUMBER>
        dispatch_instance("scalar conservation");
  } // namespace ScalarConservation
} // namespace ryujin
