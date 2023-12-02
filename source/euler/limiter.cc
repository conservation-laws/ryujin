//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#include "limiter.template.h"

using namespace dealii;

namespace ryujin
{
  namespace Euler
  {
    /* instantiations */

    template class Limiter<1, NUMBER>;
    template class Limiter<2, NUMBER>;
    template class Limiter<3, NUMBER>;

    template class Limiter<1, dealii::VectorizedArray<NUMBER>>;
    template class Limiter<2, dealii::VectorizedArray<NUMBER>>;
    template class Limiter<3, dealii::VectorizedArray<NUMBER>>;
  } // namespace Euler
} // namespace ryujin
