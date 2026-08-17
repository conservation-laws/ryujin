//
// SPDX-License-Identifier: Apache-2.0
// [LANL Copyright Statement]
// Copyright (C) 2020 - 2024 by the ryujin authors
// Copyright (C) 2023 - 2024 by Triad National Security, LLC
//

#include "wave_speed_estimator.template.h"

#include <deal.II/base/vectorization.h>

namespace ryujin
{
  namespace ShallowWater
  {
    /* instantiations */

    template class WaveSpeedEstimatorView<1, NUMBER>;
    template class WaveSpeedEstimatorView<2, NUMBER>;
    template class WaveSpeedEstimatorView<3, NUMBER>;

    template class WaveSpeedEstimatorView<1, dealii::VectorizedArray<NUMBER>>;
    template class WaveSpeedEstimatorView<2, dealii::VectorizedArray<NUMBER>>;
    template class WaveSpeedEstimatorView<3, dealii::VectorizedArray<NUMBER>>;
  } // namespace ShallowWater
} // namespace ryujin
