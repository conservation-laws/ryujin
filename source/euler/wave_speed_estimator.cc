//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#include "wave_speed_estimator.template.h"

#include <deal.II/base/vectorization.h>

namespace ryujin
{
  namespace Euler
  {
    /* instantiations */

    template class WaveSpeedEstimatorView<1, NUMBER>;
    template class WaveSpeedEstimatorView<2, NUMBER>;
    template class WaveSpeedEstimatorView<3, NUMBER>;

    template class WaveSpeedEstimatorView<1, dealii::VectorizedArray<NUMBER>>;
    template class WaveSpeedEstimatorView<2, dealii::VectorizedArray<NUMBER>>;
    template class WaveSpeedEstimatorView<3, dealii::VectorizedArray<NUMBER>>;

  } // namespace Euler
} // namespace ryujin
