//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#include "riemann_solver.template.h"

#include <deal.II/base/vectorization.h>

namespace ryujin
{
  /* instantiations */

  template class RiemannSolver<DIM, NUMBER>;
  template class RiemannSolver<DIM, VectorizedArray<NUMBER>>;

} /* namespace ryujin */
