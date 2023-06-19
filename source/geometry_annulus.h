//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "geometry_common_includes.h"

namespace ryujin
{
  namespace GridGenerator
  {
    /**
     * Create a 2D/3D partial annulus configuration with the given length,
     * height, radii and angle coverage.
     *
     * We set slip boundary conditions on all boundaries.
     *
     * The 3D mesh is created by extruding the 2D mesh with a width equal
     * to the "height".
     *
     * @ingroup Mesh
     */
    template <int dim, int spacedim, template <int, int> class Triangulation>
    void annulus(Triangulation<dim, spacedim> &,
                 const double /*length*/,
                 const double /*height*/,
                 const double /*inner_radius*/,
                 const double /*outer_radius*/,
                 const double /*angle*/)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
      __builtin_trap();
    }


#ifndef DOXYGEN
    template <template <int, int> class Triangulation>
    void annulus(Triangulation<2, 2> &triangulation,
                 const double length,
                 const double inner_radius,
                 const double outer_radius,
                 const double angle)
    {
      constexpr int dim = 2;

      using namespace dealii;

      /* Create inner ball with radius=inner_radius and rotate */
      dealii::Triangulation<dim> tria_inner;
      GridGenerator::hyper_ball(tria_inner, dealii::Point<dim>(), inner_radius);
      GridTools::rotate(dealii::numbers::PI / 4., tria_inner);

      /* Create outer annulus. Note part of this will be removed */
      dealii::Triangulation<dim> annulus;
      GridGenerator::hyper_shell(
          annulus, dealii::Point<dim>(), inner_radius, outer_radius, 4);
      GridTools::rotate(dealii::numbers::PI / 4., annulus);

      /* Create outside shell */
      dealii::Triangulation<dim> tria_outer;
      GridGenerator::hyper_shell(tria_outer,
                                 dealii::Point<dim>(),
                                 outer_radius,
                                 length / 2. * std::sqrt(2),
                                 4);

      /* Create triangulation to merge */
      dealii::Triangulation<dim, dim> final;
      final.set_mesh_smoothing(triangulation.get_mesh_smoothing());
      GridGenerator::merge_triangulations(
          {&tria_inner, &annulus, &tria_outer}, final, 1.e-12);

      /* Then, do magic to make mesh better */
      triangulation.copy_triangulation(final);

      triangulation.reset_all_manifolds();
      triangulation.set_all_manifold_ids(0);

      const auto radius_1 = inner_radius;
      const auto radius_2 = outer_radius;

      for (const auto &cell : triangulation.cell_iterators()) {
        for (const auto &face : cell->face_iterators()) {
          bool face_at_inner_sphere_boundary = true;
          bool face_at_outer_sphere_boundary = true;
          for (const auto v : face->vertex_indices()) {
            if (std::abs(face->vertex(v).norm() - radius_1) > 1.e-12)
              face_at_inner_sphere_boundary = false;
            if (std::abs(face->vertex(v).norm() - radius_2) > 1.e-12)
              face_at_outer_sphere_boundary = false;
          }
          if (face_at_inner_sphere_boundary)
            face->set_all_manifold_ids(1);
          if (face_at_outer_sphere_boundary)
            face->set_all_manifold_ids(2);
        }

        const auto position = cell->center();
        if (position.norm() < radius_1)
          cell->set_material_id(1);
        else if (position.norm() > radius_1 && position.norm() < radius_2)
          cell->set_material_id(2);
        else
          cell->set_material_id(0);
      }

      triangulation.set_manifold(1, SphericalManifold<dim>());
      triangulation.set_manifold(2, SphericalManifold<dim>());

      dealii::TransfiniteInterpolationManifold<dim> transfinite_manifold;
      transfinite_manifold.initialize(triangulation);
      triangulation.set_manifold(0, transfinite_manifold);

      triangulation.refine_global(5);
      GridTools::rotate(dealii::numbers::PI / 4., triangulation);

      /* Remove mesh cells in the annulus */

      std::set<typename dealii::Triangulation<dim>::active_cell_iterator>
          cells_to_remove;

      for (const auto &cell : triangulation.active_cell_iterators()) {
        for (auto f : dealii::GeometryInfo<dim>::face_indices()) {
          auto face = cell->face(f);
          const auto position = face->center();
          const auto radius = position.norm();
          const auto inner_value = inner_radius;
          const auto outer_value = outer_radius;

          bool in_anulus =
              radius - inner_value > 1.e-12 && outer_value - radius > 1.e-12;

          bool partial_annulus =
              std::abs(position[1]) -
                  std::abs(position[0]) *
                      std::tan(dealii::numbers::PI / 180. * angle) <
              1.e-8;

          if (in_anulus && partial_annulus) {
            cells_to_remove.insert(cell);
          }
        }
      }

      GridGenerator::create_triangulation_with_removed_cells(
          triangulation, cells_to_remove, triangulation);


      /*
       * Set boundary ids:
       */

      for (auto cell : triangulation.active_cell_iterators()) {
        for (auto f : GeometryInfo<2>::face_indices()) {
          const auto face = cell->face(f);

          if (!face->at_boundary())
            continue;

          /*
           * We want slip boundary conditions everywhere.
           */
          face->set_boundary_id(Boundary::slip);
        }
      }
    }


    template <template <int, int> class Triangulation>
    void annulus(Triangulation<3, 3> /* &triangulation */,
                 const double /* length */,
                 const double /* inner_radius */,
                 const double /* outer_radius */,
                 const double /* angle */)
    {
      using namespace dealii;
      AssertThrow(false, dealii::ExcNotImplemented());
      __builtin_trap();
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
    class Annulus : public Geometry<dim>
    {
    public:
      Annulus(const std::string subsection)
          : Geometry<dim>("annulus", subsection)
      {
        length_ = 2.;
        this->add_parameter(
            "length", length_, "length of computational domain [-L,L]x[-L,L]");

        inner_radius_ = 0.6;
        this->add_parameter(
            "inner radius", inner_radius_, "inner radius of partial annulus");

        outer_radius_ = 0.7;
        this->add_parameter(
            "outer radius", outer_radius_, "outer radius of partial annulus");

        angle_ = 45.;
        this->add_parameter("coverage angle",
                            angle_,
                            "angle coverage of partial annulus above y-axis");
      }

      void create_triangulation(
          typename Geometry<dim>::Triangulation &triangulation) final
      {
        GridGenerator::annulus(
            triangulation, length_, inner_radius_, outer_radius_, angle_);
      }

    private:
      double length_;
      double inner_radius_;
      double outer_radius_;
      double angle_;
    };
  } /* namespace Geometries */
} /* namespace ryujin */
