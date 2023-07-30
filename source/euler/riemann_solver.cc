//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#include "riemann_solver.template.h"

#include <deal.II/base/vectorization.h>

namespace ryujin
{
  namespace Euler
  {
    /* instantiations */

    template class RiemannSolver<DIM, NUMBER>;
    template class RiemannSolver<DIM, dealii::VectorizedArray<NUMBER>>;

  } // namespace Euler
} // namespace ryujin
