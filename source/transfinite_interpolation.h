//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception or LGPL-2.1-or-later
// Copyright (C) 2007 - 2022 by Martin Kronbichler
// Copyright (C) 2008 - 2022 by David Wells
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <deal.II/base/config.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/grid/reference_cell.h>
#include <deal.II/grid/tria.h>

namespace ryujin
{
  using namespace dealii; // FIXME: namespace pollution

  /**
   * A transfinite interpolation patch bound to a single coarse cell.
   *
   * In contrast to the TransfiniteInterpolationManifold shipped with
   * deal.II, this class copies all relevant geometry and manifold
   * information from a given (coarse) cell and only implements the push
   * forward: transform() maps a point of the undeformed (straight-sided)
   * coarse cell to the curved geometry.
   *
   * It is meant to be used with a MappingQCache on a triangulation that is
   * refined without any manifolds attached, see
   * Geometry::transformation().
   *
   * @ingroup Mesh
   */
  template <int dim, int spacedim = dim>
  class TransfiniteInterpolationPatch : public Manifold<dim, spacedim>
  {
  public:
    TransfiniteInterpolationPatch(
        const typename Triangulation<dim, spacedim>::cell_iterator &cell,
        const Manifold<dim, spacedim> &chart_manifold = FlatManifold<dim>());

    std::unique_ptr<Manifold<dim, spacedim>> clone() const override;

    /* Forward map of a point of the undeformed (straight-sided) coarse cell: */
    Point<spacedim> transform(const Point<spacedim> &point) const;

    /* Only defined for surrounding points that are vertices of the cell: */
    Point<spacedim>
    get_new_point(const ArrayView<const Point<spacedim>> &surrounding_points,
                  const ArrayView<const double> &weights) const override;

  private:
    Point<spacedim> push_forward(const Point<dim> &chart_point) const;

    /* We only support hypercubes, create a constexpr copy for sizing arrays: */
    static constexpr ReferenceCell<dim> reference_cell =
        ReferenceCells::get_hypercube<dim>();

    std::array<Point<spacedim>, reference_cell.n_vertices()> vertices;

    /* Shared immutable copies so that copying the patch stays cheap: */
    std::array<std::shared_ptr<const Manifold<dim, spacedim>>,
               reference_cell.n_lines()>
        line_manifolds;

    /* Only used for dim == 3: */
    std::array<std::shared_ptr<const Manifold<dim, spacedim>>,
               reference_cell.n_faces()>
        face_manifolds;

    std::shared_ptr<const Manifold<dim, spacedim>> chart_manifold;
  };

} // namespace ryujin
