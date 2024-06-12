//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#include "mesh_adaptor.template.h"
#include <instantiate.h>

namespace ryujin
{
  /* instantiations */
  template class MeshAdaptor<Description, 1, NUMBER>;
  template class MeshAdaptor<Description, 2, NUMBER>;
  template class MeshAdaptor<Description, 3, NUMBER>;

} /* namespace ryujin */
