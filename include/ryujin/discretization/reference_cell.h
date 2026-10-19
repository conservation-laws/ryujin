//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <deal.II/base/config.h>

#include <deal.II/grid/reference_cell.h>

#if !DEAL_II_VERSION_GTE(9, 8, 0)

DEAL_II_NAMESPACE_OPEN

namespace ReferenceCells
{
  template <int dim>
  inline ReferenceCell
  n_vertices_to_reference_cell(const unsigned int n_vertices)
  {
    return ReferenceCell::n_vertices_to_type(dim, n_vertices);
  }
} // namespace ReferenceCells

#if !DEAL_II_VERSION_GTE(9, 7, 0)
namespace numbers
{
  constexpr unsigned char default_geometric_orientation =
      ReferenceCell::default_combined_face_orientation();
} // namespace numbers
#endif

DEAL_II_NAMESPACE_CLOSE

#endif
