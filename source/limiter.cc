//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#include "limiter.template.h"

using namespace dealii;

namespace ryujin
{
  /* instantiations */

  template std::tuple<NUMBER, bool>
  Limiter<DIM, NUMBER>::limit(const ProblemDescription &,
                              const std::array<NUMBER, 3> &,
                              const state_type &,
                              const state_type &,
                              const NUMBER,
                              const NUMBER);

  template std::tuple<VectorizedArray<NUMBER>, bool>
  Limiter<DIM, VectorizedArray<NUMBER>>::limit(
      const ProblemDescription &,
      const std::array<VectorizedArray<NUMBER>, 3> &,
      const state_type &,
      const state_type &,
      const VectorizedArray<NUMBER>,
      const VectorizedArray<NUMBER>);

} // namespace ryujin
