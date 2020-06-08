//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#include "riemann_solver.template.h"

#include <deal.II/base/vectorization.h>

namespace ryujin
{
  /* instantiations */

#ifndef OBSESSIVE_INLINING
  template class ryujin::RiemannSolver<DIM, NUMBER>;
  template class ryujin::RiemannSolver<DIM, VectorizedArray<NUMBER>>;
#endif

} /* namespace ryujin */
