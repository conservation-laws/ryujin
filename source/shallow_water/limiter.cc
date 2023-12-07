//
// SPDX-License-Identifier: MIT or BSD-3-Clause
// [LANL Copyright Statement]
// Copyright (C) 2020 - 2023 by the ryujin authors
// Copyright (C) 2023 - 2023 by Triad National Security, LLC
//

#include "limiter.template.h"

using namespace dealii;

namespace ryujin
{
  namespace ShallowWater
  {
    /* instantiations */

    template class Limiter<1, NUMBER>;
    template class Limiter<2, NUMBER>;
    template class Limiter<3, NUMBER>;

    template class Limiter<1, dealii::VectorizedArray<NUMBER>>;
    template class Limiter<2, dealii::VectorizedArray<NUMBER>>;
    template class Limiter<3, dealii::VectorizedArray<NUMBER>>;
  } // namespace ShallowWater
} // namespace ryujin
