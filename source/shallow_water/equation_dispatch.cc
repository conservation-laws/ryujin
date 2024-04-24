//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#include "description.h"

#include <compile_time_options.h>
#include <equation_dispatch.h>

namespace ryujin
{
  namespace ShallowWater
  {
    Dispatch<Description, NUMBER> dispatch_instance("shallow water");
  } // namespace ShallowWater
} // namespace ryujin
