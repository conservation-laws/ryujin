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

    template std::tuple<NUMBER, bool>
    Limiter<DIM, NUMBER>::limit(const HyperbolicSystemView &,
                                const std::array<NUMBER, 3> &,
                                const state_type &,
                                const state_type &,
                                const NUMBER,
                                const unsigned int,
                                const NUMBER,
                                const NUMBER);

    template std::tuple<VectorizedArray<NUMBER>, bool>
    Limiter<DIM, VectorizedArray<NUMBER>>::limit(
        const HyperbolicSystemView &,
        const std::array<VectorizedArray<NUMBER>, 3> &,
        const state_type &,
        const state_type &,
        const NUMBER,
        const unsigned int,
        const VectorizedArray<NUMBER>,
        const VectorizedArray<NUMBER>);

  } // namespace Euler
} // namespace ryujin
