//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#include "hyperbolic_module.template.h"
#include <instantiate.h>

#define INSTANTIATE(dim, stages)                                               \
  template NUMBER HyperbolicModule<Description, dim, NUMBER>::step<stages>(    \
      const vector_type &,                                                     \
      std::array<std::reference_wrapper<const vector_type>, stages>,           \
      std::array<std::reference_wrapper<const precomputed_vector_type>,        \
                 stages>,                                                      \
      const std::array<NUMBER, stages>,                                        \
      vector_type &,                                                           \
      precomputed_vector_type &,                                               \
      NUMBER) const

namespace ryujin
{
  /* instantiations */

  template class HyperbolicModule<Description, 1, NUMBER>;
  template class HyperbolicModule<Description, 2, NUMBER>;
  template class HyperbolicModule<Description, 3, NUMBER>;

  INSTANTIATE(1, 0);
  INSTANTIATE(1, 1);
  INSTANTIATE(1, 2);
  INSTANTIATE(1, 3);
  INSTANTIATE(1, 4);

  INSTANTIATE(2, 0);
  INSTANTIATE(2, 1);
  INSTANTIATE(2, 2);
  INSTANTIATE(2, 3);
  INSTANTIATE(2, 4);

  INSTANTIATE(3, 0);
  INSTANTIATE(3, 1);
  INSTANTIATE(3, 2);
  INSTANTIATE(3, 3);
  INSTANTIATE(3, 4);
} /* namespace ryujin */
