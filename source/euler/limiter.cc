//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#include "limiter.template.h"

using namespace dealii;

namespace ryujin
{
  namespace Euler
  {
    /* instantiations */

    template class LimiterView<1, NUMBER>;
    template class LimiterView<2, NUMBER>;
    template class LimiterView<3, NUMBER>;

    template class LimiterView<1, dealii::VectorizedArray<NUMBER>>;
    template class LimiterView<2, dealii::VectorizedArray<NUMBER>>;
    template class LimiterView<3, dealii::VectorizedArray<NUMBER>>;
  } // namespace Euler
} // namespace ryujin
