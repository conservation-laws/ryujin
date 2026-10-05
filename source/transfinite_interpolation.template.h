//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception or LGPL-2.1-or-later
// Copyright (C) 2007 - 2022 by Martin Kronbichler
// Copyright (C) 2008 - 2022 by David Wells
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include "transfinite_interpolation.h"

#include <deal.II/fe/mapping_q_internal.h>

#include <algorithm>

namespace ryujin
{
  template <int dim, int spacedim>
  TransfiniteInterpolationPatch<dim, spacedim>::TransfiniteInterpolationPatch(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const Manifold<dim, spacedim> &chart_manifold)
      : chart_manifold(chart_manifold.clone())
  {
    AssertThrow(dim > 1, ExcNotImplemented());
    Assert(cell->reference_cell() == reference_cell,
           dealii::ExcMessage("TransfiniteInterpolationPatch requires a "
                              "quadrilateral or hexahedral cell."));

    for (const unsigned int v : reference_cell.vertex_indices())
      vertices[v] = cell->vertex(v);

    /*
     * Capture the manifolds of all lines (and faces in 3D) that differ from
     * the manifold of the cell itself. Entities without an own manifold are
     * left empty and will be interpolated from their vertices.
     */
    const auto capture = [&](const auto &entity, auto &manifold) {
      const auto id = entity->manifold_id();
      if (id != cell->manifold_id() && id != numbers::flat_manifold_id)
        manifold = cell->get_triangulation().get_manifold(id).clone();
    };
    for (const unsigned int l : reference_cell.line_indices())
      capture(cell->line(l), line_manifolds[l]);
    if constexpr (dim == 3)
      for (const unsigned int f : reference_cell.face_indices())
        capture(cell->face(f), face_manifolds[f]);
  }


  template <int dim, int spacedim>
  std::unique_ptr<Manifold<dim, spacedim>>
  TransfiniteInterpolationPatch<dim, spacedim>::clone() const
  {
    return std::make_unique<TransfiniteInterpolationPatch<dim, spacedim>>(
        *this);
  }


  template <int dim, int spacedim>
  Point<spacedim> TransfiniteInterpolationPatch<dim, spacedim>::push_forward(
      const Point<dim> &chart_point) const
  {
    Point<spacedim> new_point;

    /*
     * The transfinite interpolation adds the contribution of the faces,
     * subtracts the edges, and adds the vertices (in 3D; in 2D faces are
     * edges). If a lower dimensional entity has no own manifold, its
     * contribution is interpolated from its vertices, which we account for by
     * merging its weights into the vertex weights (with a negative sign).
     */

    if constexpr (dim == 2) {
      /*
       * Formula, see https://en.wikipedia.org/wiki/Transfinite_interpolation
       * S(u,v) = (1-v)c_1(u)+v c_3(u) + (1-u)c_2(v) + u c_4(v) -
       *   [(1-u)(1-v)P_0 + u(1-v) P_1 + (1-u)v P_2 + uv P_3]
       */
      std::array<double, 4> weights_vertices{
          {(1. - chart_point[0]) * (1. - chart_point[1]),
           chart_point[0] * (1. - chart_point[1]),
           (1. - chart_point[0]) * chart_point[1],
           chart_point[0] * chart_point[1]}};

      std::array<double, 2> weights;
      std::array<Point<spacedim>, 2> points;
      const auto weights_view = make_array_view(weights.begin(), weights.end());
      const auto points_view = make_array_view(points.begin(), points.end());

      /* Add the contribution from the lines around the cell: */
      for (const unsigned int line : reference_cell.line_indices()) {
        const double my_weight =
            (line % 2) ? chart_point[line / 2] : 1 - chart_point[line / 2];
        const double line_point = chart_point[1 - line / 2];
        const unsigned int v0 = reference_cell.line_to_cell_vertices(line, 0);
        const unsigned int v1 = reference_cell.line_to_cell_vertices(line, 1);

        if (!line_manifolds[line]) {
          weights_vertices[v0] -= my_weight * (1. - line_point);
          weights_vertices[v1] -= my_weight * line_point;
        } else {
          points = {{vertices[v0], vertices[v1]}};
          weights = {{1. - line_point, line_point}};
          new_point += my_weight * line_manifolds[line]->get_new_point(
                                       points_view, weights_view);
        }
      }

      /* Subtract contribution from the vertices (second line in formula) */
      for (const unsigned int v : reference_cell.vertex_indices())
        new_point -= weights_vertices[v] * vertices[v];

    } else {
      static_assert(dim == 3);

      /*
       * Store the components of the linear shape functions because we need
       * them repeatedly. we allow for 10 such shape functions to wrap
       * around the first four once again for easier face access.
       */
      double linear_shapes[10];
      for (unsigned int d = 0; d < 3; ++d) {
        linear_shapes[2 * d] = 1. - chart_point[d];
        linear_shapes[2 * d + 1] = chart_point[d];
      }

      /* Wrap linear shape functions around for access in face loop. */
      for (unsigned int d = 6; d < 10; ++d)
        linear_shapes[d] = linear_shapes[d - 6];

      std::array<double, 8> weights_vertices;
      for (unsigned int i2 = 0, v = 0; i2 < 2; ++i2)
        for (unsigned int i1 = 0; i1 < 2; ++i1)
          for (unsigned int i0 = 0; i0 < 2; ++i0, ++v)
            weights_vertices[v] =
                (linear_shapes[4 + i2] * linear_shapes[2 + i1]) *
                linear_shapes[i0];

      /*
       * Identify the weights for the lines to be accumulated (vertex weights
       * are set outside and coincide with the flat manifold case)
       */

      std::array<double, reference_cell.n_lines()> weights_lines;
      std::fill(weights_lines.begin(), weights_lines.end(), 0.0);

      /* Start with the contributions of the faces. */
      std::array<double, 4> weights;
      std::array<Point<spacedim>, 4> points;
      const auto weights_view = make_array_view(weights.begin(), weights.end());
      const auto points_view = make_array_view(points.begin(), points.end());

      constexpr auto orientation = numbers::default_geometric_orientation;
      for (const unsigned int face : reference_cell.face_indices()) {
        const double my_weight = linear_shapes[face];
        const unsigned int face_even = face - face % 2;

        if (std::abs(my_weight) < 1e-13)
          continue;

        std::array<unsigned int, 4> face_vertices;
        for (const unsigned int v :
             ReferenceCells::Quadrilateral.vertex_indices())
          face_vertices[v] =
              reference_cell.face_to_cell_vertices(face, v, orientation);

        if (!face_manifolds[face]) {
          /*
           * No own manifold -> face will interpolate from the surrounding lines
           * and vertices.
           */
          for (const unsigned int line :
               ReferenceCells::Quadrilateral.line_indices()) {
            const double line_weight = linear_shapes[face_even + 2 + line];
            weights_lines[reference_cell.face_to_cell_lines(
                face, line, orientation)] += my_weight * line_weight;
          }
          /*
           * As to the indices inside linear_shapes: we use the index wrapped
           * around at 2*d, ensuring the correct orientation of the face's
           * coordinate system with respect to the lexicographic indices.
           */
          weights_vertices[face_vertices[0]] -=
              linear_shapes[face_even + 2] *
              (linear_shapes[face_even + 4] * my_weight);
          weights_vertices[face_vertices[1]] -=
              linear_shapes[face_even + 3] *
              (linear_shapes[face_even + 4] * my_weight);
          weights_vertices[face_vertices[2]] -=
              linear_shapes[face_even + 2] *
              (linear_shapes[face_even + 5] * my_weight);
          weights_vertices[face_vertices[3]] -=
              linear_shapes[face_even + 3] *
              (linear_shapes[face_even + 5] * my_weight);

        } else {
          /* We have a face manifold: */

          for (const unsigned int v :
               ReferenceCells::Quadrilateral.vertex_indices())
            points[v] = vertices[face_vertices[v]];
          weights[0] =
              linear_shapes[face_even + 2] * linear_shapes[face_even + 4];
          weights[1] =
              linear_shapes[face_even + 3] * linear_shapes[face_even + 4];
          weights[2] =
              linear_shapes[face_even + 2] * linear_shapes[face_even + 5];
          weights[3] =
              linear_shapes[face_even + 3] * linear_shapes[face_even + 5];
          new_point += my_weight * face_manifolds[face]->get_new_point(
                                       points_view, weights_view);
        }
      }

      /*
       * Next subtract the contributions of the lines.
       */

      const auto weights_view_line =
          make_array_view(weights.begin(), weights.begin() + 2);
      const auto points_view_line =
          make_array_view(points.begin(), points.begin() + 2);
      for (const unsigned int line : reference_cell.line_indices()) {
        const double line_point =
            (line < 8 ? chart_point[1 - (line % 4) / 2] : chart_point[2]);
        double my_weight = 0.;
        if (line < 8)
          my_weight = linear_shapes[line % 4] * linear_shapes[4 + line / 4];
        else {
          const unsigned int subline = line - 8;
          my_weight =
              linear_shapes[subline % 2] * linear_shapes[2 + subline / 2];
        }
        my_weight -= weights_lines[line];

        if (std::abs(my_weight) < 1e-13)
          continue;

        const unsigned int v0 = reference_cell.line_to_cell_vertices(line, 0);
        const unsigned int v1 = reference_cell.line_to_cell_vertices(line, 1);

        if (!line_manifolds[line]) {
          weights_vertices[v0] -= my_weight * (1. - line_point);
          weights_vertices[v1] -= my_weight * (line_point);
        } else {
          points[0] = vertices[v0];
          points[1] = vertices[v1];
          weights[0] = 1. - line_point;
          weights[1] = line_point;
          new_point -= my_weight * line_manifolds[line]->get_new_point(
                                       points_view_line, weights_view_line);
        }
      }

      /*
       * Finally add the contribution of the vertices.
       */

      for (const unsigned int v : reference_cell.vertex_indices())
        new_point += weights_vertices[v] * vertices[v];
    }

    return new_point;
  }


  template <int dim, int spacedim>
  Point<spacedim> TransfiniteInterpolationPatch<dim, spacedim>::transform(
      const Point<spacedim> &point) const
  {
    /*
     * Compute the chart coordinates of the point with respect to the
     * undeformed (straight-sided) coarse cell by inverting its d-linear
     * map. In 2D deal.II provides a closed-form solution, in 3D we run a
     * Newton iteration starting at the center of the cell:
     */
    Point<dim> chart_point;
    if constexpr (dim == 2) {
      chart_point = dealii::internal::MappingQ1::transform_real_to_unit_cell(
          vertices, point);
    } else {
      for (unsigned int d = 0; d < dim; ++d)
        chart_point[d] = 0.5;
      for (unsigned int iteration = 0; iteration < 20; ++iteration) {
        Tensor<1, spacedim> residual = point;
        Tensor<2, dim> jacobian;
        for (const unsigned int v : reference_cell.vertex_indices()) {
          residual -= reference_cell.d_linear_shape_function(chart_point, v) *
                      vertices[v];
          jacobian += outer_product(
              vertices[v],
              reference_cell.d_linear_shape_function_gradient(chart_point, v));
        }
        const auto update = invert(jacobian) * residual;
        chart_point += update;
        if (update.norm_square() < 1.e-30)
          break;
      }
    }

    /*
     * Interpolate the unit vertices in the chart manifold with the d-linear
     * weights of the chart point. This is precisely what happens when a cell
     * is refined with this patch attached as a manifold:
     */
    std::array<Point<dim>, reference_cell.n_vertices()> unit_vertices;
    std::array<double, reference_cell.n_vertices()> weights;
    for (const unsigned int v : reference_cell.vertex_indices()) {
      unit_vertices[v] = reference_cell.vertex(v);
      weights[v] = reference_cell.d_linear_shape_function(chart_point, v);
    }
    return push_forward(chart_manifold->get_new_point(
        make_array_view(unit_vertices), make_array_view(weights)));
  }


  template <int dim, int spacedim>
  Point<spacedim> TransfiniteInterpolationPatch<dim, spacedim>::get_new_point(
      const ArrayView<const Point<spacedim>> &surrounding_points,
      const ArrayView<const double> &weights) const
  {
    /*
     * A patch is only asked for new points when it has been captured as a
     * line or face manifold of a neighboring patch. The surrounding points
     * are then vertices of the coarse cell whose chart coordinates we know:
     */
    std::array<Point<dim>, reference_cell.n_vertices()> chart_points;
    for (unsigned int i = 0; i < surrounding_points.size(); ++i) {
      const auto it =
          std::find(vertices.begin(), vertices.end(), surrounding_points[i]);
      Assert(it != vertices.end(),
             dealii::ExcMessage("TransfiniteInterpolationPatch::get_new_point "
                                "is only defined for vertices of the cell."));
      chart_points[i] =
          reference_cell.vertex(std::distance(vertices.begin(), it));
    }
    return push_forward(chart_manifold->get_new_point(
        make_array_view(chart_points.begin(),
                        chart_points.begin() + surrounding_points.size()),
        weights));
  }

} /* namespace ryujin */
