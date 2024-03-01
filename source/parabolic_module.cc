//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#include "parabolic_module.template.h"
#include <instantiate.h>

#define INSTANTIATE(dim, stages)                                               \
  template void ParabolicModule<Description, dim, NUMBER>::step<stages>(       \
      const vector_type &,                                                     \
      const NUMBER,                                                            \
      std::array<std::reference_wrapper<const vector_type>, stages>,           \
      const std::array<NUMBER, stages>,                                        \
      vector_type &,                                                           \
      const NUMBER) const

namespace ryujin
{
  /* instantiations */
  template class ParabolicModule<Description, 1, NUMBER>;
  template class ParabolicModule<Description, 2, NUMBER>;
  template class ParabolicModule<Description, 3, NUMBER>;

  INSTANTIATE(1, 0);

  INSTANTIATE(2, 0);

  INSTANTIATE(3, 0);

} /* namespace ryujin */
