//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2025 by the ryujin authors
// Copyright (C) 2025 by Triad National Security, LLC
//

#include "riemann_solver.template.h"

#include <deal.II/base/vectorization.h>

namespace ryujin
{
  namespace MultiSpeciesEuler
  {
    /* instantiations */

    template class RiemannSolver<1, NUMBER>;
    template class RiemannSolver<2, NUMBER>;
    template class RiemannSolver<3, NUMBER>;

    template class RiemannSolver<1, dealii::VectorizedArray<NUMBER>>;
    template class RiemannSolver<2, dealii::VectorizedArray<NUMBER>>;
    template class RiemannSolver<3, dealii::VectorizedArray<NUMBER>>;
  } // namespace MultiSpeciesEuler
} // namespace ryujin
