//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#include "discretization.template.h"

namespace ryujin
{
  /* instantiations */
  template class Discretization<1>;
  template class Discretization<2>;
  template class Discretization<3>;

} /* namespace ryujin */
