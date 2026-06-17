//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2025 by the ryujin authors
// Copyright (C) 2025 by Triad National Security, LLC
//

#include "description.h"

#include <compile_time_options.h>
#include <equation_dispatch.h>

namespace ryujin
{
  namespace MultiSpeciesEuler
  {
    Dispatch<Description, NUMBER> dispatch_instance("multi species euler");
  } // namespace MultiSpeciesEuler
} // namespace ryujin
