//
// SPDX-License-Identifier: Apache-2.0
// [LANL Copyright Statement]
// Copyright (C) 2020 - 2024 by the ryujin authors
// Copyright (C) 2023 - 2024 by Triad National Security, LLC
//

#include "limiter.template.h"

using namespace dealii;

namespace ryujin
{
  namespace ShallowWater
  {
    /* instantiations */

    template class LimiterView<1, NUMBER>;
    template class LimiterView<2, NUMBER>;
    template class LimiterView<3, NUMBER>;

    template class LimiterView<1, dealii::VectorizedArray<NUMBER>>;
    template class LimiterView<2, dealii::VectorizedArray<NUMBER>>;
    template class LimiterView<3, dealii::VectorizedArray<NUMBER>>;
  } // namespace ShallowWater
} // namespace ryujin
