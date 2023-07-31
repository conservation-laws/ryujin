//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "geometry_common_includes.h"

namespace ryujin
{
  namespace GridGenerator
  {
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
      __builtin_trap();
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

      dealii::Triangulation<dim, dim> tria1, tria2, tria3, tria4, tria5, tria6,
          tria7;

      GridGenerator::hyper_cube_with_cylindrical_hole(
          tria1, cylinder_diameter / 2., cylinder_diameter, 0.5, 1, false);

      GridGenerator::subdivided_hyper_rectangle(
          tria2,
          {2, 1},
          Point<2>(-cylinder_diameter, -cylinder_diameter),
          Point<2>(cylinder_diameter, -height / 2.));

      GridGenerator::subdivided_hyper_rectangle(
          tria3,
          {2, 1},
          Point<2>(-cylinder_diameter, cylinder_diameter),
          Point<2>(cylinder_diameter, height / 2.));

      GridGenerator::subdivided_hyper_rectangle(
          tria4,
          {6, 2},
          Point<2>(cylinder_diameter, -cylinder_diameter),
          Point<2>(length - cylinder_position, cylinder_diameter));

      GridGenerator::subdivided_hyper_rectangle(
          tria5,
          {6, 1},
          Point<2>(cylinder_diameter, cylinder_diameter),
          Point<2>(length - cylinder_position, height / 2.));

      GridGenerator::subdivided_hyper_rectangle(
          tria6,
          {6, 1},
          Point<2>(cylinder_diameter, -height / 2.),
          Point<2>(length - cylinder_position, -cylinder_diameter));

      tria7.set_mesh_smoothing(triangulation.get_mesh_smoothing());
      GridGenerator::merge_triangulations(
          {&tria1, &tria2, &tria3, &tria4, &tria5, &tria6},
          tria7,
          1.e-12,
          true);
      triangulation.copy_triangulation(tria7);

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
      tria2.set_mesh_smoothing(triangulation.get_mesh_smoothing());

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
  } /* namespace GridGenerator */


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

      void create_triangulation(
          typename Geometry<dim>::Triangulation &triangulation) final
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
  } /* namespace Geometries */
} /* namespace ryujin */
