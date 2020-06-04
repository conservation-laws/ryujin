//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef GRID_GENERATOR_H
#define GRID_GENERATOR_H

#include <compile_time_options.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

namespace ryujin
{
  /**
   * This namespace provides a collection of functions for generating
   * triangulations for some benchmark configurations.
   *
   * @ingroup Discretization
   */
  namespace GridGenerator
  {
    /**
     * Create a 2D triangulation consisting of a rectangle with a prescribed
     * length and height with an insribed obstacle given by a centered,
     * equilateral triangle.
     *
     * Even though this function has a template parameter @p dim, it is only
     * implemented for dimension 2.
     *
     * @ingroup Discretization
     */
    template <int dim>
    void
    coarse_grid_triangle(dealii::parallel::distributed::Triangulation<dim> &,
                         const double /*length*/,
                         const double /*height*/,
                         const double /*object_height*/)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }

#ifndef DOXYGEN
    template <>
    void coarse_grid_triangle<2>(
        dealii::parallel::distributed::Triangulation<2> &triangulation,
        const double length,
        const double height,
        const double object_height)
    {
      constexpr int dim = 2;

      using namespace dealii;

      const double object_length = object_height * std::sqrt(3) / 2.;

      const std::vector<Point<dim>> vertices{
          {0., 0.},                            // 0, bottom left
          {(length - object_length) / 2., 0.}, // 1, bottom center left
          {(length + object_length) / 2., 0.}, // 2, bottom center right
          {length, 0.},                        // 3, bottom right
          {0., height / 2.},                   // 4, middle left
          {(length - object_length) / 2., height / 2.}, // 5, middle center left
          {(length + object_length) / 2.,
           (height - object_height) / 2.}, // 6, middle lower center right
          {length, (height - object_height) / 2.}, // 7, middle lower right
          {(length + object_length) / 2.,
           (height + object_height) / 2.}, // 8, middle upper center right
          {length, (height + object_height) / 2.}, // 9, middle upper right
          {0., height},                            // 10, top left
          {(length - object_length) / 2., height}, // 11, top center left
          {(length + object_length) / 2., height}, // 12, top center right
          {length, height}                         // 13, top right
      };

      std::vector<CellData<dim>> cells(7);
      {
        const auto assign = [](auto b, std::array<unsigned int, 4> a) {
          for (unsigned int i = 0; i < 4; ++i)
            b[i] = a[i];
        };
        assign(cells[0].vertices, {0, 1, 4, 5});
        assign(cells[1].vertices, {1, 2, 5, 6});
        assign(cells[2].vertices, {2, 3, 6, 7});
        assign(cells[3].vertices, {4, 5, 10, 11});
        assign(cells[4].vertices, {5, 8, 11, 12});
        assign(cells[5].vertices, {8, 9, 12, 13});
        assign(cells[6].vertices, {6, 7, 8, 9});
      }

      triangulation.triangulation(vertices, cells, SubCellData());

      /*
       * Set boundary ids:
       */

      for (auto cell : triangulation.active_cell_iterators()) {
        for (auto f : GeometryInfo<dim>::face_indices()) {
          const auto face = cell->face(f);

          if (!face->at_boundary())
            continue;

          /*
           * We want slip boundary conditions (i.e. indicator 1) at top
           * bottom and on the triangle. On the left and right side we leave
           * the boundary indicator at 0, i.e. do nothing.
           */
          const auto center = face->center();
          if (center[0] > 0. && center[0] < length)
            face->set_boundary_id(Boundary::slip);
        }
      }
    }
#endif


    /**
     * Create an nD tube with a given length and diameter. More precisely,
     * this is a line in 1D, a rectangle in 2D, and a cylinder in 3D.
     *
     * We set boundary indicator 0 on the left and right side to indicate "do
     * nothing" boundary conditions, and boundary indicator 1 at the top and
     * bottom side in 2D, as well as the curved portion of the boundary in 3D
     * to indicate "slip boundary conditions".
     *
     * @todo Refactor prescribe/periodic into enum
     *
     * @ingroup Discretization
     */
    template <int dim>
    void coarse_grid_tube(dealii::parallel::distributed::Triangulation<dim> &,
                          const double /*length*/,
                          const double /*diameter*/,
                          const bool /*prescribe*/,
                          const bool /*periodic*/)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }


#ifndef DOXYGEN
    template <>
    void coarse_grid_tube<1>(
        dealii::parallel::distributed::Triangulation<1> &triangulation,
        const double length,
        const double /*diameter*/,
        const bool prescribe,
        const bool periodic)
    {
      dealii::GridGenerator::hyper_cube(triangulation, 0., length);
      const auto cell = triangulation.begin_active();
      if (prescribe) {
        /* Dirichlet data: */
        cell->face(0)->set_boundary_id(Boundary::dirichlet);
        cell->face(1)->set_boundary_id(Boundary::dirichlet);
      } else if (periodic) {
        /* Periodic data: */
        cell->face(0)->set_boundary_id(Boundary::periodic);
        cell->face(1)->set_boundary_id(Boundary::periodic);
      } else {
        /* Do nothing: */
        cell->face(0)->set_boundary_id(Boundary::do_nothing);
        cell->face(1)->set_boundary_id(Boundary::do_nothing);
      }
    }


    template <>
    void coarse_grid_tube<2>(
        dealii::parallel::distributed::Triangulation<2> &triangulation,
        const double length,
        const double diameter,
        const bool prescribe,
        const bool /*periodic*/)
    {
      using namespace dealii;

      GridGenerator::hyper_rectangle(triangulation,
                                     Point<2>(-length / 2., -diameter / 2.),
                                     Point<2>(length / 2., diameter / 2.));

      /*
       * Set boundary ids:
       */

      for (auto cell : triangulation.active_cell_iterators()) {
        for (auto f : GeometryInfo<2>::face_indices()) {
          const auto face = cell->face(f);

          if (!face->at_boundary())
            continue;

          if (prescribe) {
            /* Dirichlet data: */
            face->set_boundary_id(Boundary::dirichlet);

          } else {

            const auto center = face->center();
            if (center[0] < -length / 2. + 1.e-6) {

              /* Dirichlet conditions for inflow: */
              face->set_boundary_id(Boundary::dirichlet);

            } else if (center[0] > length / 2. - 1.e-6) {

              /* The right side of the domain: */
              face->set_boundary_id(Boundary::do_nothing);

            } else {

              /* Top and bottom of computational domain: */
              face->set_boundary_id(Boundary::slip);
            }
          }
        }
      }
    }


    template <>
    void coarse_grid_tube<3>(
        dealii::parallel::distributed::Triangulation<3> &triangulation,
        const double length,
        const double diameter,
        const bool prescribe,
        const bool /*periodic*/)
    {
      using namespace dealii;

      GridGenerator::cylinder(triangulation, diameter / 2., length / 2.);

      /*
       * Set boundary ids:
       */

      for (auto cell : triangulation.active_cell_iterators()) {
        for (auto f : GeometryInfo<2>::face_indices()) {
          const auto face = cell->face(f);

          if (!face->at_boundary())
            continue;

          /*
           * We want slip boundary conditions (i.e. indicator 1) at top and
           * bottom of the rectangle. On the left side we enforce initial
           * conditionas and leave the boundary indicator on the right side
           * at 0, i.e. do nothing.
           */

          if (prescribe) {
            /* Dirichlet data: */
            face->set_boundary_id(Boundary::dirichlet);

          } else {

            AssertThrow(false, dealii::ExcNotImplemented());
          }
        }
      }
    }
#endif


    /**
     * Create the 2D mach step triangulation.
     *
     * Even though this function has a template parameter @p dim, it is only
     * implemented for dimension 2.
     *
     * @ingroup Discretization
     */
    template <int dim>
    void coarse_grid_step(dealii::parallel::distributed::Triangulation<dim> &,
                          const double /*length*/,
                          const double /*height*/,
                          const double /*step_position*/,
                          const double /*step_height*/)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }


#ifndef DOXYGEN
    template <>
    void coarse_grid_step<2>(
        dealii::parallel::distributed::Triangulation<2> &triangulation,
        const double length,
        const double height,
        const double step_position,
        const double step_height)
    {
      constexpr int dim = 2;

      using namespace dealii;

      Triangulation<dim> tria1;

      GridGenerator::subdivided_hyper_rectangle(
          tria1, {15, 4}, Point<2>(0., step_height), Point<2>(length, height));

      Triangulation<dim> tria2;

      GridGenerator::subdivided_hyper_rectangle(
          tria2,
          {3, 1},
          Point<2>(0., 0.),
          Point<2>(step_position, step_height));

      GridGenerator::merge_triangulations(tria1, tria2, triangulation);

      /*
       * Set boundary ids:
       */

      for (auto cell : triangulation.active_cell_iterators()) {
        for (auto f : GeometryInfo<2>::face_indices()) {
          const auto face = cell->face(f);

          if (!face->at_boundary())
            continue;

          /*
           * We want slip boundary conditions (i.e. indicator 1) at top
           * and bottom of the rectangle. On the left we set inflow
           * conditions (i.e. indicator 2), and we do nothing on the right
           * side (i.e. indicator 0).
           */

          const auto center = face->center();

          if (center[0] > 0. + 1.e-6 && center[0] < length - 1.e-6)
            face->set_boundary_id(Boundary::slip);

          if (center[0] < 0. + 1.e-06)
            face->set_boundary_id(Boundary::dirichlet);
        }
      }

      /*
       * Refine four times and round off corner with radius 0.0125:
       */

      triangulation.refine_global(4);

      Point<dim> point(step_position + 0.0125, step_height - 0.0125);
      triangulation.set_manifold(1, SphericalManifold<dim>(point));

      for (auto cell : triangulation.active_cell_iterators())
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
             ++v) {
          double distance =
              (cell->vertex(v) - Point<dim>(step_position, step_height)).norm();
          if (distance < 1.e-6) {
            for (auto f : GeometryInfo<dim>::face_indices()) {
              const auto face = cell->face(f);
              if (face->at_boundary())
                face->set_manifold_id(Boundary::slip);
              cell->set_manifold_id(1); // temporarily for second loop
            }
          }
        }

      for (auto cell : triangulation.active_cell_iterators()) {
        if (cell->manifold_id() != 1)
          continue;

        cell->set_manifold_id(0); // reset manifold id again

        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
             ++v) {
          auto &vertex = cell->vertex(v);

          if (std::abs(vertex[0] - step_position) < 1.e-6 &&
              vertex[1] > step_height - 1.e-6)
            vertex[0] = step_position + 0.0125 * (1 - std::sqrt(1. / 2.));

          if (std::abs(vertex[1] - step_height) < 1.e-6 &&
              vertex[0] < step_position + 0.005)
            vertex[1] = step_height - 0.0125 * (1 - std::sqrt(1. / 2.));
        }
      }
    }
#endif


    /**
     * Create a 2D/3D cylinder configuration with the given length and
     * height.
     *
     * We set Dirichlet boundary conditions on the left boundary and
     * do-nothing boundary conditions on the right boundary. All other
     * boundaries (including the cylinder) have slip boundary conditions
     * prescribed.
     *
     * The 3D mesh is created by extruding the 2D mesh with a width equal
     * to the "height".
     *
     * @ingroup Discretization
     */

    template <int dim>
    void
    coarse_grid_cylinder(dealii::parallel::distributed::Triangulation<dim> &,
                         const double /*length*/,
                         const double /*height*/,
                         const double /*cylinder_position*/,
                         const double /*cylinder_height*/)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }


#ifndef DOXYGEN
    template <>
    void coarse_grid_cylinder<2>(
        dealii::parallel::distributed::Triangulation<2> &triangulation,
        const double length,
        const double height,
        const double cylinder_position,
        const double cylinder_diameter)
    {
      constexpr int dim = 2;

      using namespace dealii;

      Triangulation<dim> tria1, tria2, tria3, tria4;

      GridGenerator::hyper_cube_with_cylindrical_hole(
          tria1, cylinder_diameter / 2., cylinder_diameter, 0.5, 1, false);

      GridGenerator::subdivided_hyper_rectangle(
          tria2,
          {2, 1},
          Point<2>(-cylinder_diameter, cylinder_diameter),
          Point<2>(cylinder_diameter, height / 2.));

      GridGenerator::subdivided_hyper_rectangle(
          tria3,
          {2, 1},
          Point<2>(-cylinder_diameter, -cylinder_diameter),
          Point<2>(cylinder_diameter, -height / 2.));

      GridGenerator::subdivided_hyper_rectangle(
          tria4,
          {6, 4},
          Point<2>(cylinder_diameter, -height / 2.),
          Point<2>(length - cylinder_position, height / 2.));

      GridGenerator::merge_triangulations(
          {&tria1, &tria2, &tria3, &tria4}, triangulation, 1.e-12, true);

      /* Restore polar manifold for disc: */

      triangulation.set_manifold(0, PolarManifold<2>(Point<2>()));

      /* Fix up position of left boundary: */

      for (auto cell : triangulation.active_cell_iterators())
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
             ++v) {
          auto &vertex = cell->vertex(v);
          if (vertex[0] <= -cylinder_diameter + 1.e-6)
            vertex[0] = -cylinder_position;
        }

      /*
       * Set boundary ids:
       */

      for (auto cell : triangulation.active_cell_iterators()) {
        for (auto f : GeometryInfo<2>::face_indices()) {
          const auto face = cell->face(f);

          if (!face->at_boundary())
            continue;

          /*
           * We want slip boundary conditions (i.e. indicator 1) at top and
           * bottom of the channel, as well as on the obstacle. On the left
           * side we set inflow conditions (indicator 2) and on the right
           * side we set indicator 0, i.e. do nothing.
           */

          const auto center = face->center();

          if (center[0] > length - cylinder_position - 1.e-6) {
            face->set_boundary_id(Boundary::do_nothing);
            continue;
          }

          if (center[0] < -cylinder_position + 1.e-6) {
            face->set_boundary_id(Boundary::dirichlet);
            continue;
          }

          // the rest:
          face->set_boundary_id(Boundary::slip);
        }
      }
    }


    template <>
    void coarse_grid_cylinder<3>(
        dealii::parallel::distributed::Triangulation<3> &triangulation,
        const double length,
        const double height,
        const double cylinder_position,
        const double cylinder_diameter)
    {
      using namespace dealii;

      parallel::distributed::Triangulation<2> tria1(
          triangulation.get_communicator());

      coarse_grid_cylinder(
          tria1, length, height, cylinder_position, cylinder_diameter);

      GridGenerator::extrude_triangulation(
          tria1, 4, height, triangulation, true);
      GridTools::transform(
          [height](auto point) {
            return point - dealii::Tensor<1, 3>{{0, 0, height / 2.}};
          },
          triangulation);

      /*
       * Reattach an appropriate manifold ID:
       */

      triangulation.set_manifold(
          0, CylindricalManifold<3>(Tensor<1, 3>{{0., 0., 1.}}, Point<3>()));

      /*
       * Set boundary ids:
       */

      for (auto cell : triangulation.active_cell_iterators()) {
        for (auto f : GeometryInfo<3>::face_indices()) {
          const auto face = cell->face(f);

          if (!face->at_boundary())
            continue;

          /*
           * We want slip boundary conditions (i.e. indicator 1) almost
           * everywhere except on the faces with normal in x-direction.
           * There, on the left side we set inflow conditions (indicator 2)
           * and on the right side we set indicator 0, i.e. do nothing.
           */

          const auto center = face->center();

          if (center[0] > length - cylinder_position - 1.e-6) {
            face->set_boundary_id(Boundary::do_nothing);
            continue;
          }

          if (center[0] < -cylinder_position + 1.e-6) {
            face->set_boundary_id(Boundary::dirichlet);
            continue;
          }

          // the rest:
          face->set_boundary_id(Boundary::slip);
        }
      }
    }
#endif


    /**
     * Create a 2D double-mach reflection wall configuration:
     *
     * A rectangular domain with given length and height. The boundary
     * conditions are dirichlet conditions on the top and left of the
     * domain and do-nothing on the right side and the front part of the
     * bottom side from `[0, wall_position)`. Slip boundary conditions are
     * enforced on the bottom side on `[wall_position, length)`.
     *
     * @ingroup Discretization
     */

    template <int dim>
    void coarse_grid_wall(dealii::parallel::distributed::Triangulation<dim> &,
                          const double /*length*/,
                          const double /*height*/,
                          const double /*wall_position*/)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }


#ifndef DOXYGEN
    template <>
    void coarse_grid_wall<2>(
        dealii::parallel::distributed::Triangulation<2> &triangulation,
        const double length,
        const double height,
        const double wall_position)
    {
      using namespace dealii;

      Triangulation<2> tria1;

      GridGenerator::subdivided_hyper_rectangle(
          tria1, {18, 6}, Point<2>(wall_position, 0), Point<2>(length, height));

      Triangulation<2> tria2;

      GridGenerator::subdivided_hyper_rectangle(
          tria2, {1, 6}, Point<2>(0., 0.), Point<2>(wall_position, height));

      GridGenerator::merge_triangulations(tria1, tria2, triangulation);

      /*
       * Set boundary ids:
       */

      for (auto cell : triangulation.active_cell_iterators()) {
        for (auto f : GeometryInfo<2>::face_indices()) {
          const auto face = cell->face(f);

          if (!face->at_boundary())
            continue;

          /*
           * We want slip boundary conditions (i.e. indicator 1) at the
           * bottom starting at position wall_position. We do nothing on the
           * right boundary and enforce inflow conditions elsewhere
           */

          const auto center = face->center();

          if (center[0] > wall_position && center[1] < 1.e-6) {
            face->set_boundary_id(Boundary::slip);

          } else if (center[0] > length - 1.e-6) {

            face->set_boundary_id(Boundary::do_nothing);

          } else {

            // the rest:
            face->set_boundary_id(Boundary::dirichlet);
          }
        } /*f*/
      }   /*cell*/
    }
#endif

  } // namespace GridGenerator
} /* namespace ryujin */

#endif /* GRID_GENERATOR_H */

