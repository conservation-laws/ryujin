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
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

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
    using namespace dealii::GridGenerator;


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
     * @ingroup Discretization
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
        for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell;
             ++v) {
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

        for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell;
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
   * @ingroup Discretization
   */
  namespace Geometries
  {
    /**
     * A 2D/3D cylinder configuration constructed with
     * GridGenerator::cylinder().
     *
     * @ingroup Discretization
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
     * @ingroup Discretization
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
     * @ingroup Discretization
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
     * Describes a geometry used for validation: The (scaled) unit
     * hypercube with Dirichlet data.
     *
     * @ingroup Discretization
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
            face->set_boundary_id(Boundary::dirichlet);
          }
      }

    private:
      double length_;
    };

  } /* namespace Geometries */

} /* namespace ryujin */

#endif /* GRID_GENERATOR_H */
