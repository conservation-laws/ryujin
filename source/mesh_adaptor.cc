//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#include "mesh_adaptor.template.h"
#include <instantiate.h>

namespace ryujin
{
  /* instantiations */
  template class MeshAdaptor<HyperbolicDescription, 1, NUMBER>;
  template class MeshAdaptor<HyperbolicDescription, 2, NUMBER>;
  template class MeshAdaptor<HyperbolicDescription, 3, NUMBER>;

} /* namespace ryujin */
