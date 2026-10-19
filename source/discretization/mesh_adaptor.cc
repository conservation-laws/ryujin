//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#include "mesh_adaptor.template.h"

namespace ryujin
{
  /* instantiations */
  template class MeshAdaptor<1, NUMBER>;
  template class MeshAdaptor<2, NUMBER>;
  template class MeshAdaptor<3, NUMBER>;

} /* namespace ryujin */
