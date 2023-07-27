//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#include "description.h"
#include "parabolic_solver.template.h"

namespace ryujin
{
  namespace NavierStokes
  {
    template class ParabolicSolver<Description, DIM, NUMBER>;
  }
} // namespace ryujin

