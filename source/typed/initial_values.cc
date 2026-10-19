//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#include <instantiate.h>

#include <ryujin/typed/initial_values.template.h>

namespace ryujin
{
  /* instantiations */
  template class InitialValues<HyperbolicDescription, 1, NUMBER>;
  template class InitialValues<HyperbolicDescription, 2, NUMBER>;
  template class InitialValues<HyperbolicDescription, 3, NUMBER>;

} /* namespace ryujin */
