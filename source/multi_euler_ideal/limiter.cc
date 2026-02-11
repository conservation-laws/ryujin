//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2025 by the ryujin authors
// Copyright (C) 2025 by Triad National Security, LLC
//

#include "limiter.template.h"

using namespace dealii;

namespace ryujin
{
  namespace MultiSpeciesEuler
  {
    /* instantiations */

    template class Limiter<1, NUMBER>;
    template class Limiter<2, NUMBER>;
    template class Limiter<3, NUMBER>;

    template class Limiter<1, dealii::VectorizedArray<NUMBER>>;
    template class Limiter<2, dealii::VectorizedArray<NUMBER>>;
    template class Limiter<3, dealii::VectorizedArray<NUMBER>>;
  } // namespace MultiSpeciesEuler
} // namespace ryujin
