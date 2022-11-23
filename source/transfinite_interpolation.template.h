//
// SPDX-License-Identifier: LGPL-2.1+ or MIT
// Copyright (C) 2007 - 2022 by Martin Kronbichler
// Copyright (C) 2008 - 2022 by David Wells
//

#pragma once

#include "transfinite_interpolation.h"

#include <boost/container/small_vector.hpp>

#include <deal.II/base/table.h>

namespace ryujin
{

  namespace internal
  {
    static constexpr double invalid_pull_back_coordinate = 20.0;
  }


  template <int dim, int spacedim>
  TransfiniteInterpolationManifold<dim,
                                   spacedim>::TransfiniteInterpolationManifold()
      : level_coarse(-1)
  {
    AssertThrow(dim > 1, ExcNotImplemented());
    __builtin_trap();
  }


  template <int dim, int spacedim>
  std::unique_ptr<Manifold<dim, spacedim>>
  TransfiniteInterpolationManifold<dim, spacedim>::clone() const
  {
    auto ptr = new TransfiniteInterpolationManifold<dim, spacedim>();
    if (triangulation.n_levels() != 0)
      ptr->initialize(triangulation, *chart_manifold);
    return std::unique_ptr<Manifold<dim, spacedim>>(ptr);
  }


  template <int dim, int spacedim>
  void TransfiniteInterpolationManifold<dim, spacedim>::initialize(
      const Triangulation<dim, spacedim> &external_triangulation,
      const Manifold<dim, spacedim> &chart_manifold)
  {
    this->triangulation.clear();
    this->triangulation.copy_triangulation(external_triangulation);

    this->chart_manifold = chart_manifold.clone();

    level_coarse = triangulation.last()->level();
    coarse_cell_is_flat.resize(triangulation.n_cells(level_coarse), false);
    typename Triangulation<dim, spacedim>::active_cell_iterator
        cell = triangulation.begin(level_coarse),
        endc = triangulation.end(level_coarse);
    for (; cell != endc; ++cell) {
      bool cell_is_flat = true;
      for (unsigned int l = 0; l < GeometryInfo<dim>::lines_per_cell; ++l)
        if (cell->line(l)->manifold_id() != cell->manifold_id() &&
            cell->line(l)->manifold_id() != numbers::flat_manifold_id)
          cell_is_flat = false;
      if (dim > 2)
        for (unsigned int q = 0; q < GeometryInfo<dim>::quads_per_cell; ++q)
          if (cell->quad(q)->manifold_id() != cell->manifold_id() &&
              cell->quad(q)->manifold_id() != numbers::flat_manifold_id)
            cell_is_flat = false;
      AssertIndexRange(static_cast<unsigned int>(cell->index()),
                       coarse_cell_is_flat.size());
      coarse_cell_is_flat[cell->index()] = cell_is_flat;
    }
  }


  namespace
  {
    // version for 1D
    template <typename AccessorType>
    Point<AccessorType::space_dimension>
    compute_transfinite_interpolation(const AccessorType &cell,
                                      const Point<1> &chart_point,
                                      const bool /*cell_is_flat*/)
    {
      return cell.vertex(0) * (1. - chart_point[0]) +
             cell.vertex(1) * chart_point[0];
    }

    // version for 2D
    template <typename AccessorType>
    Point<AccessorType::space_dimension>
    compute_transfinite_interpolation(const AccessorType &cell,
                                      const Point<2> &chart_point,
                                      const bool cell_is_flat)
    {
      const unsigned int dim = AccessorType::dimension;
      const unsigned int spacedim = AccessorType::space_dimension;
      const types::manifold_id my_manifold_id = cell.manifold_id();
      const Triangulation<dim, spacedim> &tria = cell.get_triangulation();

      // formula see wikipedia
      // https://en.wikipedia.org/wiki/Transfinite_interpolation
      // S(u,v) = (1-v)c_1(u)+v c_3(u) + (1-u)c_2(v) + u c_4(v) -
      //   [(1-u)(1-v)P_0 + u(1-v) P_1 + (1-u)v P_2 + uv P_3]
      const std::array<Point<spacedim>, 4> vertices{
          {cell.vertex(0), cell.vertex(1), cell.vertex(2), cell.vertex(3)}};

      // this evaluates all bilinear shape functions because we need them
      // repeatedly. we will update this values in the complicated case with
      // curved lines below
      std::array<double, 4> weights_vertices{
          {(1. - chart_point[0]) * (1. - chart_point[1]),
           chart_point[0] * (1. - chart_point[1]),
           (1. - chart_point[0]) * chart_point[1],
           chart_point[0] * chart_point[1]}};

      Point<spacedim> new_point;
      if (cell_is_flat)
        for (const unsigned int v : GeometryInfo<2>::vertex_indices())
          new_point += weights_vertices[v] * vertices[v];
      else {
        // The second line in the formula tells us to subtract the
        // contribution of the vertices.  If a line employs the same manifold
        // as the cell, we can merge the weights of the line with the weights
        // of the vertex with a negative sign while going through the faces
        // (this is a bit artificial in 2D but it becomes clear in 3D where we
        // avoid looking at the faces' orientation and other complications).

        // add the contribution from the lines around the cell (first line in
        // formula)
        std::array<double, GeometryInfo<2>::vertices_per_face> weights;
        std::array<Point<spacedim>, GeometryInfo<2>::vertices_per_face> points;
        // note that the views are immutable, but the arrays are not
        const auto weights_view =
            make_array_view(weights.begin(), weights.end());
        const auto points_view = make_array_view(points.begin(), points.end());

        for (unsigned int line = 0; line < GeometryInfo<2>::lines_per_cell;
             ++line) {
          const double my_weight =
              (line % 2) ? chart_point[line / 2] : 1 - chart_point[line / 2];
          const double line_point = chart_point[1 - line / 2];

          // Same manifold or invalid id which will go back to the same
          // class -> contribution should be added for the final point,
          // which means that we subtract the current weight from the
          // negative weight applied to the vertex
          const types::manifold_id line_manifold_id =
              cell.line(line)->manifold_id();
          if (line_manifold_id == my_manifold_id ||
              line_manifold_id == numbers::flat_manifold_id) {
            weights_vertices[GeometryInfo<2>::line_to_cell_vertices(line, 0)] -=
                my_weight * (1. - line_point);
            weights_vertices[GeometryInfo<2>::line_to_cell_vertices(line, 1)] -=
                my_weight * line_point;
          } else {
            points[0] =
                vertices[GeometryInfo<2>::line_to_cell_vertices(line, 0)];
            points[1] =
                vertices[GeometryInfo<2>::line_to_cell_vertices(line, 1)];
            weights[0] = 1. - line_point;
            weights[1] = line_point;
            new_point +=
                my_weight * tria.get_manifold(line_manifold_id)
                                .get_new_point(points_view, weights_view);
          }
        }

        // subtract contribution from the vertices (second line in formula)
        for (const unsigned int v : GeometryInfo<2>::vertex_indices())
          new_point -= weights_vertices[v] * vertices[v];
      }

      return new_point;
    }

    // this is replicated from GeometryInfo::face_to_cell_vertices since we need
    // it very often in compute_transfinite_interpolation and the function is
    // performance critical
    static constexpr unsigned int face_to_cell_vertices_3d[6][4] = {
        {0, 2, 4, 6},
        {1, 3, 5, 7},
        {0, 4, 1, 5},
        {2, 6, 3, 7},
        {0, 1, 2, 3},
        {4, 5, 6, 7}};

    // this is replicated from GeometryInfo::face_to_cell_lines since we need it
    // very often in compute_transfinite_interpolation and the function is
    // performance critical
    static constexpr unsigned int face_to_cell_lines_3d[6][4] = {{8, 10, 0, 4},
                                                                 {9, 11, 1, 5},
                                                                 {2, 6, 8, 9},
                                                                 {3, 7, 10, 11},
                                                                 {0, 1, 2, 3},
                                                                 {4, 5, 6, 7}};

    // version for 3D
    template <typename AccessorType>
    Point<AccessorType::space_dimension>
    compute_transfinite_interpolation(const AccessorType &cell,
                                      const Point<3> &chart_point,
                                      const bool cell_is_flat)
    {
      const unsigned int dim = AccessorType::dimension;
      const unsigned int spacedim = AccessorType::space_dimension;
      const types::manifold_id my_manifold_id = cell.manifold_id();
      const Triangulation<dim, spacedim> &tria = cell.get_triangulation();

      // Same approach as in 2D, but adding the faces, subtracting the edges,
      // and adding the vertices
      const std::array<Point<spacedim>, 8> vertices{{cell.vertex(0),
                                                     cell.vertex(1),
                                                     cell.vertex(2),
                                                     cell.vertex(3),
                                                     cell.vertex(4),
                                                     cell.vertex(5),
                                                     cell.vertex(6),
                                                     cell.vertex(7)}};

      // store the components of the linear shape functions because we need them
      // repeatedly. we allow for 10 such shape functions to wrap around the
      // first four once again for easier face access.
      double linear_shapes[10];
      for (unsigned int d = 0; d < 3; ++d) {
        linear_shapes[2 * d] = 1. - chart_point[d];
        linear_shapes[2 * d + 1] = chart_point[d];
      }

      // wrap linear shape functions around for access in face loop
      for (unsigned int d = 6; d < 10; ++d)
        linear_shapes[d] = linear_shapes[d - 6];

      std::array<double, 8> weights_vertices;
      for (unsigned int i2 = 0, v = 0; i2 < 2; ++i2)
        for (unsigned int i1 = 0; i1 < 2; ++i1)
          for (unsigned int i0 = 0; i0 < 2; ++i0, ++v)
            weights_vertices[v] =
                (linear_shapes[4 + i2] * linear_shapes[2 + i1]) *
                linear_shapes[i0];

      Point<spacedim> new_point;
      if (cell_is_flat)
        for (unsigned int v = 0; v < 8; ++v)
          new_point += weights_vertices[v] * vertices[v];
      else {
        // identify the weights for the lines to be accumulated (vertex
        // weights are set outside and coincide with the flat manifold case)

        std::array<double, GeometryInfo<3>::lines_per_cell> weights_lines;
        std::fill(weights_lines.begin(), weights_lines.end(), 0.0);

        // start with the contributions of the faces
        std::array<double, GeometryInfo<2>::vertices_per_cell> weights;
        std::array<Point<spacedim>, GeometryInfo<2>::vertices_per_cell> points;
        // note that the views are immutable, but the arrays are not
        const auto weights_view =
            make_array_view(weights.begin(), weights.end());
        const auto points_view = make_array_view(points.begin(), points.end());

        for (const unsigned int face : GeometryInfo<3>::face_indices()) {
          const double my_weight = linear_shapes[face];
          const unsigned int face_even = face - face % 2;

          if (std::abs(my_weight) < 1e-13)
            continue;

          // same manifold or invalid id which will go back to the same class
          // -> face will interpolate from the surrounding lines and vertices
          const types::manifold_id face_manifold_id =
              cell.face(face)->manifold_id();
          if (face_manifold_id == my_manifold_id ||
              face_manifold_id == numbers::flat_manifold_id) {
            for (unsigned int line = 0; line < GeometryInfo<2>::lines_per_cell;
                 ++line) {
              const double line_weight = linear_shapes[face_even + 2 + line];
              weights_lines[face_to_cell_lines_3d[face][line]] +=
                  my_weight * line_weight;
            }
            // as to the indices inside linear_shapes: we use the index
            // wrapped around at 2*d, ensuring the correct orientation of
            // the face's coordinate system with respect to the
            // lexicographic indices
            weights_vertices[face_to_cell_vertices_3d[face][0]] -=
                linear_shapes[face_even + 2] *
                (linear_shapes[face_even + 4] * my_weight);
            weights_vertices[face_to_cell_vertices_3d[face][1]] -=
                linear_shapes[face_even + 3] *
                (linear_shapes[face_even + 4] * my_weight);
            weights_vertices[face_to_cell_vertices_3d[face][2]] -=
                linear_shapes[face_even + 2] *
                (linear_shapes[face_even + 5] * my_weight);
            weights_vertices[face_to_cell_vertices_3d[face][3]] -=
                linear_shapes[face_even + 3] *
                (linear_shapes[face_even + 5] * my_weight);
          } else {
            for (const unsigned int v : GeometryInfo<2>::vertex_indices())
              points[v] = vertices[face_to_cell_vertices_3d[face][v]];
            weights[0] =
                linear_shapes[face_even + 2] * linear_shapes[face_even + 4];
            weights[1] =
                linear_shapes[face_even + 3] * linear_shapes[face_even + 4];
            weights[2] =
                linear_shapes[face_even + 2] * linear_shapes[face_even + 5];
            weights[3] =
                linear_shapes[face_even + 3] * linear_shapes[face_even + 5];
            new_point +=
                my_weight * tria.get_manifold(face_manifold_id)
                                .get_new_point(points_view, weights_view);
          }
        }

        // next subtract the contributions of the lines
        const auto weights_view_line =
            make_array_view(weights.begin(), weights.begin() + 2);
        const auto points_view_line =
            make_array_view(points.begin(), points.begin() + 2);
        for (unsigned int line = 0; line < GeometryInfo<3>::lines_per_cell;
             ++line) {
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

          const types::manifold_id line_manifold_id =
              cell.line(line)->manifold_id();
          if (line_manifold_id == my_manifold_id ||
              line_manifold_id == numbers::flat_manifold_id) {
            weights_vertices[GeometryInfo<3>::line_to_cell_vertices(line, 0)] -=
                my_weight * (1. - line_point);
            weights_vertices[GeometryInfo<3>::line_to_cell_vertices(line, 1)] -=
                my_weight * (line_point);
          } else {
            points[0] =
                vertices[GeometryInfo<3>::line_to_cell_vertices(line, 0)];
            points[1] =
                vertices[GeometryInfo<3>::line_to_cell_vertices(line, 1)];
            weights[0] = 1. - line_point;
            weights[1] = line_point;
            new_point -= my_weight * tria.get_manifold(line_manifold_id)
                                         .get_new_point(points_view_line,
                                                        weights_view_line);
          }
        }

        // finally add the contribution of the
        for (const unsigned int v : GeometryInfo<dim>::vertex_indices())
          new_point += weights_vertices[v] * vertices[v];
      }
      return new_point;
    }
  } // namespace


  template <int dim, int spacedim>
  Point<spacedim> TransfiniteInterpolationManifold<dim, spacedim>::push_forward(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const Point<dim> &chart_point) const
  {
    AssertDimension(cell->level(), level_coarse);

    // check that the point is in the unit cell which is the current chart
    // Tolerance 5e-4 chosen that the method also works with manifolds
    // that have some discretization error like SphericalManifold
    Assert(GeometryInfo<dim>::is_inside_unit_cell(chart_point, 5e-4),
           ExcMessage("chart_point is not in unit interval"));

    return compute_transfinite_interpolation(
        *cell, chart_point, coarse_cell_is_flat[cell->index()]);
  }


  template <int dim, int spacedim>
  DerivativeForm<1, dim, spacedim>
  TransfiniteInterpolationManifold<dim, spacedim>::push_forward_gradient(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const Point<dim> &chart_point,
      const Point<spacedim> &pushed_forward_chart_point) const
  {
    // compute the derivative with the help of finite differences
    DerivativeForm<1, dim, spacedim> grad;
    for (unsigned int d = 0; d < dim; ++d) {
      Point<dim> modified = chart_point;
      const double step = chart_point[d] > 0.5 ? -1e-8 : 1e-8;

      // avoid checking outside of the unit interval
      modified[d] += step;
      Tensor<1, spacedim> difference =
          compute_transfinite_interpolation(
              *cell, modified, coarse_cell_is_flat[cell->index()]) -
          pushed_forward_chart_point;
      for (unsigned int e = 0; e < spacedim; ++e)
        grad[e][d] = difference[e] / step;
    }
    return grad;
  }


  template <int dim, int spacedim>
  Point<dim> TransfiniteInterpolationManifold<dim, spacedim>::pull_back(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const Point<spacedim> &point,
      const Point<dim> &initial_guess) const
  {
    Point<dim> outside;
    for (unsigned int d = 0; d < dim; ++d)
      outside[d] = internal::invalid_pull_back_coordinate;

    // project the user-given input to unit cell
    Point<dim> chart_point =
        GeometryInfo<dim>::project_to_unit_cell(initial_guess);

    // run quasi-Newton iteration with a combination of finite differences for
    // the exact Jacobian and "Broyden's good method". As opposed to the various
    // mapping implementations, this class does not throw exception upon failure
    // as those are relatively expensive and failure occurs quite regularly in
    // the implementation of the compute_chart_points method.
    Tensor<1, spacedim> residual =
        point - compute_transfinite_interpolation(
                    *cell, chart_point, coarse_cell_is_flat[cell->index()]);
    const double tolerance =
        1e-21 * Utilities::fixed_power<2>(cell->diameter());
    double residual_norm_square = residual.norm_square();
    DerivativeForm<1, dim, spacedim> inv_grad;
    bool must_recompute_jacobian = true;
    for (unsigned int i = 0; i < 100; ++i) {
      if (residual_norm_square < tolerance) {
        // do a final update of the point with the last available Jacobian
        // information. The residual is close to zero due to the check
        // above, but me might improve some of the last digits by a final
        // Newton-like step with step length 1
        Tensor<1, dim> update;
        for (unsigned int d = 0; d < spacedim; ++d)
          for (unsigned int e = 0; e < dim; ++e)
            update[e] += inv_grad[d][e] * residual[d];
        return chart_point + update;
      }

      // every 9 iterations, including the first time around, we create an
      // approximation of the Jacobian with finite differences. Broyden's
      // method usually does not need more than 5-8 iterations, but sometimes
      // we might have had a bad initial guess and then we can accelerate
      // convergence considerably with getting the actual Jacobian rather than
      // using secant-like methods (one gradient calculation in 3D costs as
      // much as 3 more iterations). this usually happens close to convergence
      // and one more step with the finite-differenced Jacobian leads to
      // convergence. however, we should not make the update too close to
      // termination either because of cancellation effects an finite
      // difference accuracy.
      if (must_recompute_jacobian ||
          (residual_norm_square > 1e4 * tolerance && i % 7 == 0)) {
        // if the determinant is zero or negative, the mapping is either not
        // invertible or already has inverted and we are outside the valid
        // chart region. Note that the Jacobian here represents the
        // derivative of the forward map and should have a positive
        // determinant since we use properly oriented meshes.
        DerivativeForm<1, dim, spacedim> grad = push_forward_gradient(
            cell, chart_point, Point<spacedim>(point - residual));
        if (grad.determinant() <= 0.0)
          return outside;
        inv_grad = grad.covariant_form();
        must_recompute_jacobian = false;
      }
      Tensor<1, dim> update;
      for (unsigned int d = 0; d < spacedim; ++d)
        for (unsigned int e = 0; e < dim; ++e)
          update[e] += inv_grad[d][e] * residual[d];

      // Line search, accept step if the residual has decreased
      double alpha = 1.;

      // check if point is inside 1.2 times the unit cell to avoid
      // hitting points very far away from valid ones in the manifolds
      while (!GeometryInfo<dim>::is_inside_unit_cell(
                 chart_point + alpha * update, 0.2) &&
             alpha > 1e-7)
        alpha *= 0.5;

      const Tensor<1, spacedim> old_residual = residual;
      while (alpha > 1e-4) {
        Point<dim> guess = chart_point + alpha * update;
        const Tensor<1, dim> residual_guess =
            point - compute_transfinite_interpolation(
                        *cell, guess, coarse_cell_is_flat[cell->index()]);
        const double residual_norm_new = residual_guess.norm_square();
        if (residual_norm_new < residual_norm_square) {
          residual_norm_square = residual_norm_new;
          chart_point += alpha * update;
          residual = residual_guess;
          break;
        } else
          alpha *= 0.5;
      }
      // If alpha got very small, it is likely due to a bad Jacobian
      // approximation with Broyden's method (relatively far away from the
      // zero), which can be corrected by the outer loop when a Newton update
      // is recomputed. The second case is when we either have roundoff errors
      // and cannot further improve the approximation or the Jacobian is
      // actually bad and we should fail as early as possible. Since we cannot
      // really distinguish the two, we must continue here in any case.
      if (alpha <= 1e-4) {
        // If we just recomputed the Jacobian by finite differences, we must
        // stop. If the reached tolerance was sufficiently small (less than
        // the square root of the tolerance), we return the best estimate,
        // else we return the invalid point.
        if (must_recompute_jacobian == true) {
          return residual_norm_square < std::sqrt(tolerance) ? chart_point
                                                             : outside;
        } else
          must_recompute_jacobian = true;
      }

      // update the inverse Jacobian with "Broyden's good method" and
      // Sherman-Morrison formula for the update of the inverse, see
      // https://en.wikipedia.org/wiki/Broyden%27s_method
      // J^{-1}_n = J^{-1}_{n-1} + (delta x_n - J^{-1}_{n-1} delta f_n) /
      // (delta x_n^T J_{-1}_{n-1} delta f_n) delta x_n^T J^{-1}_{n-1}

      // switch sign in residual as compared to the formula above because we
      // use a negative definition of the residual with respect to the
      // Jacobian
      const Tensor<1, spacedim> delta_f = old_residual - residual;

      Tensor<1, dim> Jinv_deltaf;
      for (unsigned int d = 0; d < spacedim; ++d)
        for (unsigned int e = 0; e < dim; ++e)
          Jinv_deltaf[e] += inv_grad[d][e] * delta_f[d];

      const Tensor<1, dim> delta_x = alpha * update;

      // prevent division by zero. This number should be scale-invariant
      // because Jinv_deltaf carries no units and x is in reference
      // coordinates.
      if (std::abs(delta_x * Jinv_deltaf) > 0.1 * tolerance &&
          !must_recompute_jacobian) {
        const Tensor<1, dim> factor =
            (delta_x - Jinv_deltaf) / (delta_x * Jinv_deltaf);
        Tensor<1, spacedim> jac_update;
        for (unsigned int d = 0; d < spacedim; ++d)
          for (unsigned int e = 0; e < dim; ++e)
            jac_update[d] += delta_x[e] * inv_grad[d][e];
        for (unsigned int d = 0; d < spacedim; ++d)
          for (unsigned int e = 0; e < dim; ++e)
            inv_grad[d][e] += factor[e] * jac_update[d];
      }
    }
    return outside;
  }


  template <int dim, int spacedim>
  std::array<unsigned int, 20> TransfiniteInterpolationManifold<dim, spacedim>::
      get_possible_cells_around_points(
          const ArrayView<const Point<spacedim>> &points) const
  {
    // The methods to identify cells around points in GridTools are all written
    // for the active cells, but we are here looking at some cells at the coarse
    // level.

    // FIXME IMPROVE

    Assert(triangulation.n_levels() != 0, ExcNotInitialized());
    Assert(triangulation.begin_active()->level() == level_coarse,
           ExcInternalError());

    // This computes the distance of the surrounding points transformed to the
    // unit cell from the unit cell.
    auto cell = triangulation.begin(level_coarse);
    const auto endc = triangulation.end(level_coarse);
    boost::container::small_vector<std::pair<double, unsigned int>, 200>
        distances_and_cells;
    for (; cell != endc; ++cell) {
      /* FIXME: Remove workaround - ignore certain cells. */
      if (cell->material_id() == 42)
        continue;

      std::array<Point<spacedim>, GeometryInfo<dim>::vertices_per_cell>
          vertices;
      for (const unsigned int vertex_n : GeometryInfo<dim>::vertex_indices()) {
        vertices[vertex_n] = cell->vertex(vertex_n);
      }

      // cheap check: if any of the points is not inside a circle around the
      // center of the loop, we can skip the expensive part below (this assumes
      // that the manifold does not deform the grid too much)
      Point<spacedim> center;
      for (const unsigned int v : GeometryInfo<dim>::vertex_indices())
        center += vertices[v];
      center *= 1. / GeometryInfo<dim>::vertices_per_cell;
      double radius_square = 0.;
      for (const unsigned int v : GeometryInfo<dim>::vertex_indices())
        radius_square =
            std::max(radius_square, (center - vertices[v]).norm_square());
      bool inside_circle = true;
      for (unsigned int i = 0; i < points.size(); ++i)
        if ((center - points[i]).norm_square() > radius_square * 1.5) {
          inside_circle = false;
          break;
        }
      if (inside_circle == false)
        continue;

      // slightly more expensive search
      double current_distance = 0;
      for (unsigned int i = 0; i < points.size(); ++i) {
        Point<dim> point =
            cell->real_to_unit_cell_affine_approximation(points[i]);
        current_distance += GeometryInfo<dim>::distance_to_unit_cell(point);
      }
      distances_and_cells.push_back(
          std::make_pair(current_distance, cell->index()));
    }
    // no coarse cell could be found -> transformation failed
    AssertThrow(distances_and_cells.size() > 0,
                (typename Mapping<dim, spacedim>::ExcTransformationFailed()));
    std::sort(distances_and_cells.begin(), distances_and_cells.end());
    std::array<unsigned int, 20> cells;
    cells.fill(numbers::invalid_unsigned_int);
    for (unsigned int i = 0; i < distances_and_cells.size() && i < cells.size();
         ++i)
      cells[i] = distances_and_cells[i].second;

    return cells;
  }


  template <int dim, int spacedim>
  typename Triangulation<dim, spacedim>::cell_iterator
  TransfiniteInterpolationManifold<dim, spacedim>::compute_chart_points(
      const ArrayView<const Point<spacedim>> &surrounding_points,
      ArrayView<Point<dim>> chart_points) const
  {
    Assert(surrounding_points.size() == chart_points.size(),
           ExcMessage("The chart points array view must be as large as the "
                      "surrounding points array view."));

    std::array<unsigned int, 20> nearby_cells =
        get_possible_cells_around_points(surrounding_points);

    // This function is nearly always called to place new points on a cell or
    // cell face. In this case, the general structure of the surrounding points
    // is known (i.e., if there are eight surrounding points, then they will
    // almost surely be either eight points around a quadrilateral or the eight
    // vertices of a cube). Hence, making this assumption, we use two
    // optimizations (one for structdim == 2 and one for structdim == 3) that
    // guess the locations of some of the chart points more efficiently than the
    // affine map approximation. The affine map approximation is used whenever
    // we don't have a cheaper guess available.

    // Function that can guess the location of a chart point by assuming that
    // the eight surrounding points are points on a two-dimensional object
    // (either a cell in 2D or the face of a hexahedron in 3D), arranged like
    //
    //     2 - 7 - 3
    //     |       |
    //     4       5
    //     |       |
    //     0 - 6 - 1
    //
    // This function assumes that the first three chart points have been
    // computed since there is no effective way to guess them.
    auto guess_chart_point_structdim_2 =
        [&](const unsigned int i) -> Point<dim> {
      Assert(
          surrounding_points.size() == 8 && 2 < i && i < 8,
          ExcMessage("This function assumes that there are eight surrounding "
                     "points around a two-dimensional object. It also assumes "
                     "that the first three chart points have already been "
                     "computed."));
      switch (i) {
      case 0:
      case 1:
      case 2:
        Assert(false, ExcInternalError());
        break;
      case 3:
        return chart_points[1] + (chart_points[2] - chart_points[0]);
      case 4:
        return 0.5 * (chart_points[0] + chart_points[2]);
      case 5:
        return 0.5 * (chart_points[1] + chart_points[3]);
      case 6:
        return 0.5 * (chart_points[0] + chart_points[1]);
      case 7:
        return 0.5 * (chart_points[2] + chart_points[3]);
      default:
        Assert(false, ExcInternalError());
      }

      return Point<dim>();
    };

    // Function that can guess the location of a chart point by assuming that
    // the eight surrounding points form the vertices of a hexahedron, arranged
    // like
    //
    //         6-------7
    //        /|      /|
    //       /       / |
    //      /  |    /  |
    //     4-------5   |
    //     |   2- -|- -3
    //     |  /    |  /
    //     |       | /
    //     |/      |/
    //     0-------1
    //
    // (where vertex 2 is the back left vertex) we can estimate where chart
    // points 5 - 7 are by computing the height (in chart coordinates) as c4 -
    // c0 and then adding that onto the appropriate bottom vertex.
    //
    // This function assumes that the first five chart points have been computed
    // since there is no effective way to guess them.
    auto guess_chart_point_structdim_3 =
        [&](const unsigned int i) -> Point<dim> {
      Assert(
          surrounding_points.size() == 8 && 4 < i && i < 8,
          ExcMessage("This function assumes that there are eight surrounding "
                     "points around a three-dimensional object. It also "
                     "assumes that the first five chart points have already "
                     "been computed."));
      return chart_points[i - 4] + (chart_points[4] - chart_points[0]);
    };

    // Check if we can use the two chart point shortcuts above before we start:
    bool use_structdim_2_guesses = false;
    bool use_structdim_3_guesses = false;
    // note that in the structdim 2 case: 0 - 6 and 2 - 7 should be roughly
    // parallel, while in the structdim 3 case, 0 - 6 and 2 - 7 should be
    // roughly orthogonal. Use the angle between these two vectors to figure out
    // if we should turn on either structdim optimization.
    if (surrounding_points.size() == 8) {
      const Tensor<1, spacedim> v06 =
          surrounding_points[6] - surrounding_points[0];
      const Tensor<1, spacedim> v27 =
          surrounding_points[7] - surrounding_points[2];

      // note that we can save a call to sqrt() by rearranging
      const double cosine = scalar_product(v06, v27) /
                            std::sqrt(v06.norm_square() * v27.norm_square());
      if (0.707 < cosine)
        // the angle is less than pi/4, so these vectors are roughly parallel:
        // enable the structdim 2 optimization
        use_structdim_2_guesses = true;
      else if (spacedim == 3)
        // otherwise these vectors are roughly orthogonal: enable the
        // structdim 3 optimization if we are in 3D
        use_structdim_3_guesses = true;
    }
    // we should enable at most one of the optimizations
    Assert((!use_structdim_2_guesses && !use_structdim_3_guesses) ||
               (use_structdim_2_guesses ^ use_structdim_3_guesses),
           ExcInternalError());


    auto compute_chart_point =
        [&](const typename Triangulation<dim, spacedim>::cell_iterator &cell,
            const unsigned int point_index) {
          Point<dim> guess;
          // an optimization: keep track of whether or not we used the affine
          // approximation so that we don't call pull_back with the same
          // initial guess twice (i.e., if pull_back fails the first time,
          // don't try again with the same function arguments).
          bool used_affine_approximation = false;
          // if we have already computed three points, we can guess the fourth
          // to be the missing corner point of a rectangle
          if (point_index == 3 && surrounding_points.size() >= 8)
            guess = chart_points[1] + (chart_points[2] - chart_points[0]);
          else if (use_structdim_2_guesses && 3 < point_index)
            guess = guess_chart_point_structdim_2(point_index);
          else if (use_structdim_3_guesses && 4 < point_index)
            guess = guess_chart_point_structdim_3(point_index);
          else if (dim == 3 && point_index > 7 &&
                   surrounding_points.size() == 26) {
            if (point_index < 20)
              guess =
                  0.5 * (chart_points[GeometryInfo<dim>::line_to_cell_vertices(
                             point_index - 8, 0)] +
                         chart_points[GeometryInfo<dim>::line_to_cell_vertices(
                             point_index - 8, 1)]);
            else
              guess =
                  0.25 * (chart_points[GeometryInfo<dim>::face_to_cell_vertices(
                              point_index - 20, 0)] +
                          chart_points[GeometryInfo<dim>::face_to_cell_vertices(
                              point_index - 20, 1)] +
                          chart_points[GeometryInfo<dim>::face_to_cell_vertices(
                              point_index - 20, 2)] +
                          chart_points[GeometryInfo<dim>::face_to_cell_vertices(
                              point_index - 20, 3)]);
          } else {
            guess = cell->real_to_unit_cell_affine_approximation(
                surrounding_points[point_index]);
            used_affine_approximation = true;
          }
          chart_points[point_index] =
              pull_back(cell, surrounding_points[point_index], guess);

          // the initial guess may not have been good enough: if applicable,
          // try again with the affine approximation (which is more accurate
          // than the cheap methods used above)
          if (chart_points[point_index][0] ==
                  internal::invalid_pull_back_coordinate &&
              !used_affine_approximation) {
            guess = cell->real_to_unit_cell_affine_approximation(
                surrounding_points[point_index]);
            chart_points[point_index] =
                pull_back(cell, surrounding_points[point_index], guess);
          }

          if (chart_points[point_index][0] ==
              internal::invalid_pull_back_coordinate) {
            for (unsigned int d = 0; d < dim; ++d)
              guess[d] = 0.5;
            chart_points[point_index] =
                pull_back(cell, surrounding_points[point_index], guess);
          }
        };

    // check whether all points are inside the unit cell of the current chart
    for (unsigned int c = 0; c < nearby_cells.size(); ++c) {
      typename Triangulation<dim, spacedim>::cell_iterator cell(
          &triangulation, level_coarse, nearby_cells[c]);
      bool inside_unit_cell = true;
      for (unsigned int i = 0; i < surrounding_points.size(); ++i) {
        compute_chart_point(cell, i);

        // Tolerance 5e-4 chosen that the method also works with manifolds
        // that have some discretization error like SphericalManifold
        if (GeometryInfo<dim>::is_inside_unit_cell(chart_points[i], 5e-4) ==
            false) {
          inside_unit_cell = false;
          break;
        }
      }
      if (inside_unit_cell == true) {
        return cell;
      }

      // if we did not find a point and this was the last valid cell (the next
      // iterate being the end of the array or an invalid tag), we must stop
      if (c == nearby_cells.size() - 1 ||
          nearby_cells[c + 1] == numbers::invalid_unsigned_int) {
        // generate additional information to help debugging why we did not
        // get a point
        std::ostringstream message;
        for (unsigned int b = 0; b <= c; ++b) {
          typename Triangulation<dim, spacedim>::cell_iterator cell(
              &triangulation, level_coarse, nearby_cells[b]);
          message << "Looking at cell " << cell->id()
                  << " with vertices: " << std::endl;
          for (const unsigned int v : GeometryInfo<dim>::vertex_indices())
            message << std::setprecision(16) << " " << cell->vertex(v)
                    << "    ";
          message << std::endl;
          message << "Transformation to chart coordinates: " << std::endl;
          for (unsigned int i = 0; i < surrounding_points.size(); ++i) {
            compute_chart_point(cell, i);
            message << std::setprecision(16) << surrounding_points[i] << " -> "
                    << chart_points[i] << std::endl;
          }
        }

        AssertThrow(false,
                    (typename Mapping<dim, spacedim>::ExcTransformationFailed(
                        message.str())));
      }
    }

    // a valid inversion should have returned a point above. an invalid
    // inversion should have triggered the assertion, so we should never end up
    // here
    Assert(false, ExcInternalError());
    return typename Triangulation<dim, spacedim>::cell_iterator();
  }


  template <int dim, int spacedim>
  Point<spacedim>
  TransfiniteInterpolationManifold<dim, spacedim>::get_new_point(
      const ArrayView<const Point<spacedim>> &surrounding_points,
      const ArrayView<const double> &weights) const
  {
    boost::container::small_vector<Point<dim>, 100> chart_points(
        surrounding_points.size());
    ArrayView<Point<dim>> chart_points_view =
        make_array_view(chart_points.begin(), chart_points.end());
    const auto cell =
        compute_chart_points(surrounding_points, chart_points_view);

    const Point<dim> p_chart =
        chart_manifold->get_new_point(chart_points_view, weights);

    return push_forward(cell, p_chart);
  }


  template <int dim, int spacedim>
  void TransfiniteInterpolationManifold<dim, spacedim>::get_new_points(
      const ArrayView<const Point<spacedim>> &surrounding_points,
      const Table<2, double> &weights,
      ArrayView<Point<spacedim>> new_points) const
  {
    Assert(weights.size(0) > 0, ExcEmptyObject());
    AssertDimension(surrounding_points.size(), weights.size(1));

    boost::container::small_vector<Point<dim>, 100> chart_points(
        surrounding_points.size());
    ArrayView<Point<dim>> chart_points_view =
        make_array_view(chart_points.begin(), chart_points.end());
    const auto cell =
        compute_chart_points(surrounding_points, chart_points_view);

    boost::container::small_vector<Point<dim>, 100> new_points_on_chart(
        weights.size(0));
    chart_manifold->get_new_points(chart_points_view,
                                   weights,
                                   make_array_view(new_points_on_chart.begin(),
                                                   new_points_on_chart.end()));

    for (unsigned int row = 0; row < weights.size(0); ++row)
      new_points[row] = push_forward(cell, new_points_on_chart[row]);
  }

} // namespace ryujin
