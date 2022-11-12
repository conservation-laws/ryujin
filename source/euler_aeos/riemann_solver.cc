//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#include "riemann_solver.template.h"

#include <deal.II/base/vectorization.h>

namespace ryujin
{
  namespace EulerAEOS
  {
    /* instantiations */

    template class RiemannSolver<DIM, NUMBER>;
    template class RiemannSolver<DIM, dealii::VectorizedArray<NUMBER>>;
  } // namespace EulerAEOS
} // namespace ryujin
