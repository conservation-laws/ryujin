//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef GRID_GENERATOR_H
#define GRID_GENERATOR_H

#include <compile_time_options.h>

#include "geometry.h"

#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

namespace ryujin
{

  namespace Manifolds
  {
    /**
     * @todo Documentation
     */
    template <int dim>
    class AirfoilManifold : public dealii::ChartManifold<dim>
    {
    public:
      AirfoilManifold(const dealii::Point<dim> airfoil_center,
                      const std::function<double(const double)> &psi_front,
                      const std::function<double(const double)> &psi_upper,
                      const std::function<double(const double)> &psi_lower,
                      const bool upper_side)
          : airfoil_center(airfoil_center)
          , psi_front(psi_front)
          , psi_upper(psi_upper)
          , psi_lower(psi_lower)
          , upper_side(upper_side)
          , polar_manifold()
      {
        Assert(std::abs(psi_upper(0.) - psi_front(0.5 * M_PI)) < 1.0e-10,
               dealii::ExcInternalError());
        Assert(std::abs(psi_lower(0.) + psi_front(1.5 * M_PI)) < 1.0e-10,
               dealii::ExcInternalError());
      }

      virtual dealii::Point<dim>
      pull_back(const dealii::Point<dim> &space_point) const final override
      {
        auto coordinate = dealii::Point<dim>() + (space_point - airfoil_center);

        /* transform: */

        dealii::Point<dim> chart_point;
        if (coordinate[0] > 0.) {
          if (upper_side) {
            /* upper back airfoil part */
            chart_point[0] = 1. + coordinate[1] - psi_upper(coordinate[0]);
            chart_point[1] = 0.5 * M_PI - coordinate[0];
          } else {
            /* lower back airfoil part */
            chart_point[0] = 1. - coordinate[1] + psi_lower(coordinate[0]);
            chart_point[1] = 1.5 * M_PI + coordinate[0];
          }
        } else {
          /* front part */
          chart_point = polar_manifold.pull_back(coordinate);
          chart_point[0] = 1. + chart_point[0] - psi_front(chart_point[1]);
        }

        return chart_point;
      }

      virtual dealii::Point<dim>
      push_forward(const dealii::Point<dim> &point) const final override
      {
        auto chart_point = point;

        /* transform back */

        dealii::Point<dim> coordinate;
        if (chart_point[1] < 0.5 * M_PI) {
          Assert(upper_side, dealii::ExcInternalError());
          /* upper back airfoil part */
          coordinate[0] = 0.5 * M_PI - chart_point[1];
          Assert(coordinate[0] >= -1.0e-10, dealii::ExcInternalError());
          coordinate[1] = chart_point[0] - 1. + psi_upper(coordinate[0]);
        } else if (chart_point[1] > 1.5 * M_PI) {
          Assert(!upper_side, dealii::ExcInternalError());
          /* lower back airfoil part */
          coordinate[0] = chart_point[1] - 1.5 * M_PI;
          Assert(coordinate[0] >= -1.0e-10, dealii::ExcInternalError());
          coordinate[1] = 1. - chart_point[0] + psi_lower(coordinate[0]);
        } else {
          /* front part */
          chart_point[0] = chart_point[0] - 1. + psi_front(chart_point[1]);
          coordinate = polar_manifold.push_forward(chart_point);
        }

        return dealii::Point<dim>() + (coordinate + airfoil_center);
      }

      std::unique_ptr<dealii::Manifold<dim, dim>> clone() const final override
      {
        return std::make_unique<AirfoilManifold<dim>>(
            airfoil_center, psi_front, psi_upper, psi_lower, upper_side);
      }

    private:
      const dealii::Point<dim> airfoil_center;
      const std::function<double(const double)> psi_front;
      const std::function<double(const double)> psi_upper;
      const std::function<double(const double)> psi_lower;
      const bool upper_side;

      dealii::PolarManifold<dim> polar_manifold;
    };


    /**
     * @todo Documentation
     */
    template <int dim>
    class GradingManifold : public dealii::ChartManifold<dim>
    {
    public:
      GradingManifold(const dealii::Point<dim> center,
                      const double grading,
                      const double epsilon)
          : center(center)
          , grading(grading)
          , epsilon(epsilon)
          , polar_manifold(center)
      {
      }

      virtual dealii::Point<dim>
      pull_back(const dealii::Point<dim> &space_point) const final override
      {
        auto point = polar_manifold.pull_back(space_point);
        Assert(point[0] >= 0., dealii::ExcInternalError());
        point[0] = std::pow(point[0] + epsilon, 1. / grading) -
                   std::pow(epsilon, 1. / grading) + 1.e-14;
        const auto chart_point = polar_manifold.push_forward(point);
        return chart_point;
      }

      virtual dealii::Point<dim>
      push_forward(const dealii::Point<dim> &chart_point) const final override
      {
        auto point = polar_manifold.pull_back(chart_point);
        point[0] =
            std::pow(point[0] + std::pow(epsilon, 1. / grading), grading) -
            epsilon + 1.e-14;
        Assert(point[0] >= 0., dealii::ExcInternalError());
        return polar_manifold.push_forward(point);
      }

      std::unique_ptr<dealii::Manifold<dim, dim>> clone() const final override
      {
        return std::make_unique<GradingManifold<dim>>(center, grading, epsilon);
      }

    private:
      const dealii::Point<dim> center;
      const double grading;
      const double epsilon;

      dealii::PolarManifold<dim> polar_manifold;
    };
  } // namespace Manifolds

  /**
   * This namespace provides a collection of functions for generating
   * triangulations for some benchmark configurations.
   *
   * @ingroup Mesh
   */
  namespace GridGenerator
  {
    using namespace dealii::GridGenerator;


    /**
     * Create a 2D airfoil
     *
     * @todo documentation
     *
     * @ingroup Mesh
     */
    template <int dim, int spacedim, template <int, int> class Triangulation>
    void airfoil(Triangulation<dim, spacedim> &,
                 const dealii::Point<spacedim> &,
                 const std::function<double(const double)> &,
                 const std::function<double(const double)> &,
                 const std::function<double(const double)> &,
                 const double,
                 const double)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }


#ifndef DOXYGEN
    template <template <int, int> class Triangulation>
    void airfoil(Triangulation<2, 2> &triangulation,
                 const dealii::Point<2> &airfoil_center,
                 const std::function<double(const double)> &psi_front,
                 const std::function<double(const double)> &psi_upper,
                 const std::function<double(const double)> &psi_lower,
                 const double outer_radius,
                 const double grading,
                 const double grading_epsilon,
                 unsigned int n_anisotropic_refinements)
    {
      /* by convention, psi_front(0.) returns the "back length" */
      const auto back_length = psi_front(0.);

      /* sharp trailing edge? */
      const bool sharp_trailing_edge =
          std::abs(psi_upper(back_length) - psi_lower(back_length)) < 1.0e-10;
      AssertThrow(
          sharp_trailing_edge ||
              std::abs(psi_upper(back_length) - psi_lower(back_length)) >
                  0.01 * back_length,
          dealii::ExcMessage("Blunt trailing edge has a width of less than "
                             "1% of the trailing airfoil length."));

      /* Front part: */

      std::vector<dealii::Point<2>> vertices1{
          {-outer_radius, 0.0},                                      // 0
          {airfoil_center[0] - psi_front(M_PI), airfoil_center[1]},  // 1
          {-0.5 * outer_radius, -std::sqrt(3.) / 2. * outer_radius}, // 2
          {0.5 * outer_radius, -std::sqrt(3.) / 2. * outer_radius},  // 3
          {airfoil_center[0], airfoil_center[1] + psi_lower(0)},     // 4
          {airfoil_center[0] + back_length,                          //
           airfoil_center[1] + psi_lower(back_length)},              // 5
          {airfoil_center[0], airfoil_center[1] + psi_upper(0)},     // 6
          {-0.5 * outer_radius, std::sqrt(3.) / 2. * outer_radius},  // 7
          {0.5 * outer_radius, std::sqrt(3.) / 2. * outer_radius},   // 8
      };

      std::vector<dealii::CellData<2>> cells1(4);
      cells1[0].vertices = {2, 3, 4, 5};
      cells1[1].vertices = {0, 2, 1, 4};
      cells1[2].vertices = {7, 0, 6, 1};
      if (sharp_trailing_edge) {
        cells1[3].vertices = {8, 7, 5, 6};
      } else {
        vertices1.push_back({airfoil_center[0] + back_length,
                             airfoil_center[1] + psi_upper(back_length)});
        cells1[3].vertices = {8, 7, 9, 6};
      }

      dealii::Triangulation<2> tria1;
      tria1.create_triangulation(vertices1, cells1, dealii::SubCellData());

      dealii::Triangulation<2> tria2;

      if (sharp_trailing_edge) {
        /* Back part for sharp trailing edge: */

        const std::vector<dealii::Point<2>> vertices2{
            {0.5 * outer_radius, -std::sqrt(3.) / 2. * outer_radius}, // 0
            {airfoil_center[0] + back_length,
             airfoil_center[1] + psi_lower(back_length)},            // 1
            {0.5 * outer_radius, std::sqrt(3.) / 2. * outer_radius}, // 2
            {outer_radius, -0.5 * outer_radius},                     // 3
            {outer_radius, 0.0},                                     // 4
            {outer_radius, 0.5 * outer_radius},                      // 5
        };

        std::vector<dealii::CellData<2>> cells2(2);
        cells2[0].vertices = {0, 3, 1, 4};
        cells2[1].vertices = {1, 4, 2, 5};

        tria2.create_triangulation(vertices2, cells2, dealii::SubCellData());

      } else {
        /* Back part for blunt trailing edge: */

        /* Good width for the anisotropically refined center trailing cell: */
        double trailing_height =
            0.5 / (0.5 + std::pow(2., n_anisotropic_refinements)) * 0.5 *
            outer_radius;

        const std::vector<dealii::Point<2>> vertices2{
            {0.5 * outer_radius, -std::sqrt(3.) / 2. * outer_radius}, // 0
            {airfoil_center[0] + back_length,
             airfoil_center[1] + psi_lower(back_length)}, // 1
            {airfoil_center[0] + back_length,
             airfoil_center[1] + psi_upper(back_length)},            // 2
            {0.5 * outer_radius, std::sqrt(3.) / 2. * outer_radius}, // 3
            {outer_radius, -0.5 * outer_radius},                     // 4
            {outer_radius, -trailing_height},                        // 5
            {outer_radius, trailing_height},                         // 6
            {outer_radius, 0.5 * outer_radius},                      // 7
        };

        std::vector<dealii::CellData<2>> cells2(3);
        cells2[0].vertices = {0, 4, 1, 5};
        cells2[1].vertices = {1, 5, 2, 6};
        cells2[2].vertices = {2, 6, 3, 7};

        tria2.create_triangulation(vertices2, cells2, dealii::SubCellData());
      }

      dealii::Triangulation<2> tria3;
      GridGenerator::merge_triangulations(
          {&tria1, &tria2}, tria3, 1.e-12, true);

      /*
       * Helper lambda to set manifold IDs and attach manifolds:
       *
       *   1 -> upper airfoil (inner boundary)
       *   2 -> lower airfoil (inner boundary)
       *   3 -> spherical manifold (outer boundary)
       *   4 -> grading front (interior face)
       *   5 -> grading upper airfoil (interior face)
       *   6 -> grading lower airfoil (interior face)
       *   7 -> grading upper back (interior face)
       *   8 -> grading lower back (interior face)
       *   9 -> transfinite interpolation
       */
      const auto attach_manifolds = [&](auto &triangulation) {
        triangulation.set_all_manifold_ids(9);

        /* all possible vertices for the four (or six) radials: */
        const std::vector<dealii::Point<2>> radial_vertices{
            {airfoil_center[0] - psi_front(M_PI), airfoil_center[1]}, // front
            {airfoil_center[0], airfoil_center[1] + psi_upper(0)},    // upper
            {airfoil_center[0], airfoil_center[1] + psi_lower(0)},    // lower
            {airfoil_center[0] + back_length,
             airfoil_center[1] + psi_upper(back_length)}, // upper back
            {airfoil_center[0] + back_length,
             airfoil_center[1] + psi_lower(back_length)}, // lower back
        };

        for (auto cell : triangulation.active_cell_iterators()) {

          for (auto f : dealii::GeometryInfo<2>::face_indices()) {
            const auto face = cell->face(f);

            if (face->at_boundary()) {
              /* Handle boundary faces: */

              bool airfoil = true;
              bool spherical_boundary = true;

              for (const auto v : dealii::GeometryInfo<1>::vertex_indices())
                if (std::abs((face->vertex(v)).norm() - outer_radius) < 1.0e-10)
                  airfoil = false;
                else
                  spherical_boundary = false;

              if (spherical_boundary) {
                face->set_manifold_id(3);
              }

              if (airfoil) {
                if (face->center()[0] <
                    airfoil_center[0] + back_length - 1.e-6) {
                  if (face->center()[1] >= airfoil_center[1]) {
                    face->set_manifold_id(1);
                  } else {
                    face->set_manifold_id(2);
                  }
                }
              }

            } else {
              /* Handle radial faces: */

              unsigned int index = 4;
              for (auto candidate : radial_vertices) {
                const auto direction_1 = candidate - face->vertex(0);
                const auto direction_2 = face->vertex(1) - face->vertex(0);
                if (direction_1.norm() < 1.0e-10 ||
                    std::abs(cross_product_2d(direction_1) * direction_2) <
                        1.0e-10) {
                  Assert(index < 10, dealii::ExcInternalError());
                  face->set_manifold_id(index);
                  break;
                }
                index++;
              }
            }
          } /* f */
        }   /* cell */

        Manifolds::AirfoilManifold airfoil_manifold_upper{
            airfoil_center, psi_front, psi_upper, psi_lower, true};
        triangulation.set_manifold(1, airfoil_manifold_upper);

        Manifolds::AirfoilManifold airfoil_manifold_lower{
            airfoil_center, psi_front, psi_upper, psi_lower, false};
        triangulation.set_manifold(2, airfoil_manifold_lower);

        dealii::SphericalManifold<2> spherical_manifold;
        triangulation.set_manifold(3, spherical_manifold);

        unsigned int index = 4;
        for (auto vertex : radial_vertices) {
          Manifolds::GradingManifold manifold{vertex, grading, grading_epsilon};
          triangulation.set_manifold(index, manifold);
          index++;
        }
        Assert(index == 9, dealii::ExcInternalError());

        dealii::TransfiniteInterpolationManifold<2> transfinite;
        transfinite.initialize(triangulation);
        triangulation.set_manifold(9, transfinite);
      };

      attach_manifolds(tria3);

      /* Anisotropic pre refinement: */

      if (sharp_trailing_edge) {
        Assert(n_anisotropic_refinements == 0, dealii::ExcNotImplemented());

      } else {

        /* mark critical cell: */
        for (auto cell : tria3.active_cell_iterators())
          if (cell->center()[0] > airfoil_center[0] + back_length &&
              std::abs(cell->center()[1]) <=
                  1.1 * std::abs(airfoil_center[1]) + 1.0e-6)
            cell->set_material_id(2);

        for (unsigned int i = 0; i < n_anisotropic_refinements; ++i) {
          for (auto cell : tria3.active_cell_iterators())
            if (cell->material_id() == 2)
              cell->set_refine_flag(dealii::RefinementCase<2>::cut_axis(0));
            else
              cell->set_refine_flag();

          tria3.execute_coarsening_and_refinement();
        }
      }

      /* Flatten triangulation and create distributed coarse triangulation: */

      dealii::Triangulation<2> tria4;
      GridGenerator::flatten_triangulation(tria3, tria4);
      triangulation.copy_triangulation(tria4);
      /* We have to re-attach manifolds after flattening and copying: */
      attach_manifolds(triangulation);

      /* Set boundary ids: */

      // FIXME
      for (auto cell : triangulation.active_cell_iterators()) {
        for (auto f : dealii::GeometryInfo<2>::face_indices()) {
          const auto face = cell->face(f);
          if (face->at_boundary()) {
            /* Handle boundary faces: */

            bool airfoil = true;
            bool spherical_boundary = true;

            for (const auto v : dealii::GeometryInfo<1>::vertex_indices())
              if (std::abs((face->vertex(v)).norm() - outer_radius) < 1.0e-10)
                airfoil = false;
              else
                spherical_boundary = false;

            if (spherical_boundary)
              face->set_boundary_id(Boundary::dirichlet);

            if (airfoil)
              face->set_boundary_id(Boundary::no_slip);
          }
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
     * @ingroup Mesh
     */

    template <int dim, int spacedim, template <int, int> class Triangulation>
    void cylinder(Triangulation<dim, spacedim> &,
                  const double /*length*/,
                  const double /*height*/,
                  const double /*cylinder_position*/,
                  const double /*cylinder_height*/)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }


#ifndef DOXYGEN
    template <template <int, int> class Triangulation>
    void cylinder(Triangulation<2, 2> &triangulation,
                  const double length,
                  const double height,
                  const double cylinder_position,
                  const double cylinder_diameter)
    {
      constexpr int dim = 2;

      using namespace dealii;

      dealii::Triangulation<dim, dim> tria1, tria2, tria3, tria4, tria5;

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
          {&tria1, &tria2, &tria3, &tria4}, tria5, 1.e-12, true);
      triangulation.copy_triangulation(tria5);

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


    template <template <int, int> class Triangulation>
    void cylinder(Triangulation<3, 3> &triangulation,
                  const double length,
                  const double height,
                  const double cylinder_position,
                  const double cylinder_diameter)
    {
      using namespace dealii;

      dealii::Triangulation<2, 2> tria1;

      cylinder(tria1, length, height, cylinder_position, cylinder_diameter);

      dealii::Triangulation<3, 3> tria2;

      GridGenerator::extrude_triangulation(tria1, 4, height, tria2, true);
      GridTools::transform(
          [height](auto point) {
            return point - dealii::Tensor<1, 3>{{0, 0, height / 2.}};
          },
          tria2);

      triangulation.copy_triangulation(tria2);

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
     * Create the 2D mach step triangulation.
     *
     * Even though this function has a template parameter @p dim, it is only
     * implemented for dimension 2.
     *
     * @ingroup Mesh
     */
    template <int dim, int spacedim, template <int, int> class Triangulation>
    void step(Triangulation<dim, dim> &,
              const double /*length*/,
              const double /*height*/,
              const double /*step_position*/,
              const double /*step_height*/)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }


#ifndef DOXYGEN
    template <template <int, int> class Triangulation>
    void step(Triangulation<2, 2> &triangulation,
              const double length,
              const double height,
              const double step_position,
              const double step_height)
    {
      using namespace dealii;

      dealii::Triangulation<2, 2> tria1, tria2, tria3;

      GridGenerator::subdivided_hyper_rectangle(
          tria1, {15, 4}, Point<2>(0., step_height), Point<2>(length, height));

      GridGenerator::subdivided_hyper_rectangle(
          tria2,
          {3, 1},
          Point<2>(0., 0.),
          Point<2>(step_position, step_height));

      GridGenerator::merge_triangulations(tria1, tria2, tria3);
      triangulation.copy_triangulation(tria3);

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

      Point<2> point(step_position + 0.0125, step_height - 0.0125);
      triangulation.set_manifold(1, SphericalManifold<2>(point));

      for (auto cell : triangulation.active_cell_iterators())
        for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v) {
          double distance =
              (cell->vertex(v) - Point<2>(step_position, step_height)).norm();
          if (distance < 1.e-6) {
            for (auto f : GeometryInfo<2>::face_indices()) {
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

        for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v) {
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
     * Create a 2D double-mach reflection wall configuration:
     *
     * A rectangular domain with given length and height. The boundary
     * conditions are dirichlet conditions on the top and left of the
     * domain and do-nothing on the right side and the front part of the
     * bottom side from `[0, wall_position)`. Slip boundary conditions are
     * enforced on the bottom side on `[wall_position, length)`.
     *
     * @ingroup Mesh
     */

    template <int dim, int spacedim, template <int, int> class Triangulation>
    void wall(Triangulation<dim, spacedim> &,
              const double /*length*/,
              const double /*height*/,
              const double /*wall_position*/)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }


#ifndef DOXYGEN
    template <template <int, int> class Triangulation>
    void wall(Triangulation<2, 2> &triangulation,
              const double length,
              const double height,
              const double wall_position)
    {
      using namespace dealii;

      dealii::Triangulation<2, 2> tria1, tria2, tria3;

      GridGenerator::subdivided_hyper_rectangle(
          tria1, {18, 6}, Point<2>(wall_position, 0), Point<2>(length, height));

      GridGenerator::subdivided_hyper_rectangle(
          tria2, {1, 6}, Point<2>(0., 0.), Point<2>(wall_position, height));

      GridGenerator::merge_triangulations(tria1, tria2, tria3);

      triangulation.copy_triangulation(tria3);

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

  } /* namespace GridGenerator */

  /**
   * A namespace for a number of benchmark geometries and dealii::GridIn
   * wrappers.
   *
   * @ingroup Mesh
   */
  namespace Geometries
  {
    /**
     * A 2D/3D cylinder configuration constructed with
     * GridGenerator::cylinder().
     *
     * @ingroup Mesh
     */
    template <int dim>
    class Cylinder : public Geometry<dim>
    {
    public:
      Cylinder(const std::string subsection)
          : Geometry<dim>("cylinder", subsection)
      {
        length_ = 4.;
        this->add_parameter(
            "length", length_, "length of computational domain");

        height_ = 2.;
        this->add_parameter(
            "height", height_, "height of computational domain");

        object_position_ = 0.6;
        this->add_parameter("object position",
                            object_position_,
                            "x position of immersed cylinder center point");

        object_diameter_ = 0.5;
        this->add_parameter("object diameter",
                            object_diameter_,
                            "diameter of immersed cylinder");
      }

      virtual void create_triangulation(
          typename Geometry<dim>::Triangulation &triangulation) final override
      {
        GridGenerator::cylinder(triangulation,
                                length_,
                                height_,
                                object_position_,
                                object_diameter_);
      }

    private:
      double length_;
      double height_;
      double object_position_;
      double object_diameter_;
    };


    /**
     * A 2D Mach step configuration constructed with GridGenerator::step().
     *
     * @ingroup Mesh
     */
    template <int dim>
    class Step : public Geometry<dim>
    {
    public:
      Step(const std::string subsection)
          : Geometry<dim>("step", subsection)
      {
        length_ = 3.;
        this->add_parameter(
            "length", length_, "length of computational domain");

        height_ = 1.;
        this->add_parameter(
            "height", height_, "height of computational domain");

        step_position_ = 0.6;
        this->add_parameter(
            "step position", step_position_, "x position of step");

        step_height_ = 0.2;
        this->add_parameter("step height", step_height_, "height of step");
      }

      virtual void create_triangulation(
          typename Geometry<dim>::Triangulation &triangulation) final override
      {
        GridGenerator::step<dim, dim>(
            triangulation, length_, height_, step_position_, step_height_);
      }

    private:
      double length_;
      double height_;
      double step_position_;
      double step_height_;
    };


    /**
     * A 2D Mach step configuration constructed with GridGenerator::step().
     *
     * @ingroup Mesh
     */
    template <int dim>
    class Wall : public Geometry<dim>
    {
    public:
      Wall(const std::string subsection)
          : Geometry<dim>("wall", subsection)
      {
        length_ = 3.2;
        this->add_parameter(
            "length", length_, "length of computational domain");

        height_ = 1.0;
        this->add_parameter(
            "height", height_, "height of computational domain");

        wall_position_ = 1. / 6.;
        this->add_parameter(
            "wall position", wall_position_, "x position of wall");
      }

      virtual void create_triangulation(
          typename Geometry<dim>::Triangulation &triangulation) final override
      {
        GridGenerator::wall(triangulation, length_, height_, wall_position_);
      }

    private:
      double length_;
      double height_;
      double wall_position_;
    };


    /**
     * A 2D/3D shocktube configuration for generating a viscous boundary
     * layer for the compressible Navier Stokes equations.
     *
     * A rectangular domain with given length and height. The boundary
     * conditions are slip boundary conditions on the top, and no_slip
     * boundary conditions on bottom, left and right boundary of the 2D
     * domain.
     *
     * @ingroup Mesh
     */
    template <int dim>
    class ShockTube : public Geometry<dim>
    {
    public:
      ShockTube(const std::string subsection)
          : Geometry<dim>("shocktube", subsection)
      {
        length_ = 1.0;
        this->add_parameter(
            "length", length_, "length of computational domain");

        height_ = 0.5;
        this->add_parameter(
            "height", height_, "height of computational domain");

        subdivisions_x_ = 2;
        this->add_parameter("subdivisions x",
                            subdivisions_x_,
                            "number of subdivisions in x direction");

        subdivisions_y_ = 1;
        this->add_parameter("subdivisions y",
                            subdivisions_y_,
                            "number of subdivisions in y direction");

        grading_push_forward_ = dim == 2 ? "x;y" : "x;y;z;";
        this->add_parameter("grading push forward",
                            grading_push_forward_,
                            "push forward of grading manifold");

        grading_pull_back_ = dim == 2 ? "x;y" : "x;y;z;";
        this->add_parameter("grading pull back",
                            grading_pull_back_,
                            "pull back of grading manifold");
      }

      virtual void create_triangulation(
          typename Geometry<dim>::Triangulation &triangulation) final override
      {
        /* create mesh: */

        dealii::Triangulation<dim, dim> tria1;
        if constexpr (dim == 2)
          dealii::GridGenerator::subdivided_hyper_rectangle(
              tria1,
              {subdivisions_x_, subdivisions_y_},
              dealii::Point<2>(),
              dealii::Point<2>(length_, height_));
        else if constexpr (dim == 3)
          dealii::GridGenerator::subdivided_hyper_rectangle(
              tria1,
              {subdivisions_x_, subdivisions_y_, subdivisions_y_},
              dealii::Point<3>(),
              dealii::Point<3>(length_, height_, height_));
        triangulation.copy_triangulation(tria1);

        /* create grading: */

        if (grading_push_forward_ != (dim == 2 ? "x;y" : "x;y;z;")) {
          dealii::FunctionManifold<dim> grading(grading_push_forward_,
                                                grading_pull_back_);
          triangulation.set_all_manifold_ids(1);
          triangulation.set_manifold(1, grading);
        }

        /* set boundary ids: */

        for (auto cell : triangulation.active_cell_iterators()) {
          for (auto f : dealii::GeometryInfo<dim>::face_indices()) {
            const auto face = cell->face(f);
            if (!face->at_boundary())
              continue;

            const auto position = face->center();
            if (position[0] < 1.e-6) {
              /* left: no slip */
              face->set_boundary_id(Boundary::no_slip);
            } else if (position[0] > length_ - 1.e-6) {
              /* right: no slip */
              face->set_boundary_id(Boundary::no_slip);
            } else if (position[1] < 1.e-6) {
              /* bottom: no slip */
              face->set_boundary_id(Boundary::no_slip);
            } else if (position[1] > height_ - 1.e-6) {
              /* top: slip */
              face->set_boundary_id(Boundary::slip);
            } else {
              if constexpr (dim == 3) {
                if (position[2] < 1.e-6) {
                  /* left: no slip */
                  face->set_boundary_id(Boundary::no_slip);
                } else if (position[2] > height_ - 1.e-6) {
                  /* right: slip */
                  face->set_boundary_id(Boundary::slip);
                } else {
                  Assert(false, dealii::ExcInternalError());
                }
              } else {
                Assert(false, dealii::ExcInternalError());
              }
            }
          } /*for*/
        }   /*for*/
      }

    private:
      double length_;
      double height_;
      unsigned int subdivisions_x_;
      unsigned int subdivisions_y_;
      std::string grading_push_forward_;
      std::string grading_pull_back_;
    };


    /**
     * A square (or hypercube) domain used for running validation
     * configurations. Per default Dirichlet boundary conditions are
     * enforced throughout. If the @ref periodic_ parameter is set to true
     * periodic boundary conditions are enforced in the y (and z)
     * directions instead.
     *
     * @ingroup Mesh
     */
    template <int dim>
    class Validation : public Geometry<dim>
    {
    public:
      Validation(const std::string subsection)
          : Geometry<dim>("validation", subsection)
      {
        length_ = 20.;
        this->add_parameter(
            "length", length_, "length of computational domain");

        periodic_ = false;
        this->add_parameter("periodic",
                            periodic_,
                            "enforce periodicity in y (and z) directions "
                            "instead of Dirichlet conditions");
      }

      virtual void create_triangulation(
          typename Geometry<dim>::Triangulation &triangulation) final override
      {
        dealii::Triangulation<dim, dim> tria1;
        dealii::GridGenerator::hyper_cube(tria1, -0.5 * length_, 0.5 * length_);
        triangulation.copy_triangulation(tria1);

        for (auto cell : triangulation.active_cell_iterators())
          for (auto f : dealii::GeometryInfo<dim>::face_indices()) {
            const auto face = cell->face(f);
            if (!face->at_boundary())
              continue;
            const auto position = face->center();
            if (position[0] < -0.5 * length_ + 1.e-6)
              /* left: dirichlet */
              face->set_boundary_id(Boundary::dirichlet);
            else if (position[0] > 0.5 * length_ - 1.e-6)
              /* right: dirichlet */
              face->set_boundary_id(Boundary::dirichlet);
            else {
              if (periodic_)
                face->set_boundary_id(Boundary::periodic);
              else
                face->set_boundary_id(Boundary::dirichlet);
            }
          }
      }

    private:
      double length_;
      bool periodic_;
    };


    /**
     * A 2D Airfoil
     *
     * @todo Description
     *
     * @ingroup Mesh
     */
    template <int dim>
    class Airfoil : public Geometry<dim>
    {
    public:
      Airfoil(const std::string subsection)
          : Geometry<dim>("airfoil", subsection)
      {
        airfoil_center_[0] = -.5;
        this->add_parameter(
            "airfoil center", airfoil_center_, "center position of airfoil");

        airfoil_length_ = 1.;
        this->add_parameter("airfoil length",
                            airfoil_length_,
                            "length of airfoil (leading to trailing edge)");

        length_ = 5.;
        this->add_parameter(
            "length", length_, "length of computational domain (diameter)");

        grading_ = 5.;
        this->add_parameter(
            "grading exponent", grading_, "graded mesh: exponent");

        grading_epsilon_ = 0.03;
        this->add_parameter("grading epsilon",
                            grading_epsilon_,
                            "graded mesh: regularization parameter");

        n_anisotropic_refinements_ = 0;
        this->add_parameter("anisotropic pre refinement",
                            n_anisotropic_refinements_,
                            "number of anisotropic pre refinement steps");
      }

      virtual void create_triangulation(
          typename Geometry<dim>::Triangulation &triangulation) final override
      {
        /* FIXME: */

        const double front_radius = 0.1 * airfoil_length_;
        const double back_length = 0.9 * airfoil_length_;

        const auto psi_front = [=](const double phi) {
          if (std::abs(phi) < 1.0e-10)
            return back_length;
          const auto a = 1.0 * front_radius;
          const auto b = front_radius;
          const auto r = a * b /
                         std::sqrt((a * std::cos(phi)) * (a * std::cos(phi)) +
                                   (b * std::sin(phi)) * (b * std::sin(phi)));
          return r;
        };

        const double bluntness = 0.005;

        const auto psi_upper = [=](const double x) {
          if (x > back_length)
            return bluntness;
          return 1.0 * front_radius + (bluntness - 1.0 * front_radius) * x * x /
                                          back_length / back_length;
        };

        const auto psi_lower = [=](const double x) {
          if (x > back_length)
            return -bluntness;
          return -1.0 * front_radius - (bluntness - 1.0 * front_radius) * x *
                                           x / back_length / back_length;
        };

        GridGenerator::airfoil(triangulation,
                               airfoil_center_,
                               psi_front,
                               psi_upper,
                               psi_lower,
                               0.5 * length_,
                               grading_,
                               grading_epsilon_,
                               n_anisotropic_refinements_);
      }

    private:
      dealii::Point<dim> airfoil_center_;
      double airfoil_length_;
      double length_;
      double grading_;
      double grading_epsilon_;
      unsigned int n_anisotropic_refinements_;
    };

  } /* namespace Geometries */

} /* namespace ryujin */

#endif /* GRID_GENERATOR_H */
