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
    template class ParabolicSolver<Description, 1, NUMBER>;
    template class ParabolicSolver<Description, 2, NUMBER>;
    template class ParabolicSolver<Description, 3, NUMBER>;
  } // namespace NavierStokes
} // namespace ryujin
