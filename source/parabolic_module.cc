//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#include "parabolic_module.template.h"
#include <instantiate.h>

#define INSTANTIATE(dim, stages)                                               \
  template void ParabolicModule<Description, dim, NUMBER>::step<stages>(       \
      const StateVector &,                                                     \
      const NUMBER,                                                            \
      std::array<std::reference_wrapper<const StateVector>, stages>,           \
      const std::array<NUMBER, stages>,                                        \
      StateVector &,                                                           \
      NUMBER) const

namespace ryujin
{
  /* instantiations */
  template class ParabolicModule<Description, 1, NUMBER>;
  template class ParabolicModule<Description, 2, NUMBER>;
  template class ParabolicModule<Description, 3, NUMBER>;

  INSTANTIATE(1, 0);
  INSTANTIATE(1, 1);
  INSTANTIATE(1, 2);
  INSTANTIATE(1, 3);

  INSTANTIATE(2, 0);
  INSTANTIATE(2, 1);
  INSTANTIATE(2, 2);
  INSTANTIATE(2, 3);

  INSTANTIATE(3, 0);
  INSTANTIATE(3, 1);
  INSTANTIATE(3, 2);
  INSTANTIATE(3, 3);

} /* namespace ryujin */
