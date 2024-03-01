//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#include "offline_data.template.h"

namespace ryujin
{
  /* instantiations */
  template class OfflineData<1, NUMBER>;
  template class OfflineData<2, NUMBER>;
  template class OfflineData<3, NUMBER>;

} /* namespace ryujin */
