//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#include "description.h"

#include <compile_time_options.h>
#include <equation_dispatch.h>

namespace ryujin
{
  namespace Skeleton
  {
    Dispatch<Description, NUMBER> dispatch_instance("skeleton");
  } // namespace Skeleton
} // namespace ryujin
