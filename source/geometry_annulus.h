//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 by the ryujin authors
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
     * @ingroup Mesh
     */
    template <int dim, int spacedim, template <int, int> class Triangulation>
    void annulus(Triangulation<dim, spacedim> &,
                 const double /*length*/,
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
      constexpr double eps = 1.0e-10;

      using namespace dealii;

      /*
       * A lambda that conveniently sets and assigns manifold IDs and that
       * we need a couple of times during the construction process of the
       * triangulation:
       */
      const auto assign_manifolds = [&](auto &tria) {
        for (const auto &cell : tria.cell_iterators()) {
          /*
           * Mark all cells that will comprise the annulus. That is, all
           * cells whose vertices lie betwenn the two radii.
           */
          bool cell_on_annulus = true;
          for (unsigned int v : cell->vertex_indices()) {
            const auto vertex = cell->vertex(v);
            const auto distance = vertex.norm();
            if (!(inner_radius - eps <= distance && distance <= outer_radius)) {
              cell_on_annulus = false;
              break;
            }
          }
          if (cell_on_annulus)
            cell->set_all_manifold_ids(1);

          /*
           * Separately, mark all faces that touch the annulus.
           */
          for (const unsigned int f : cell->face_indices()) {
            const auto face = cell->face(f);

            bool face_on_annulus = true;
            for (unsigned int v : face->vertex_indices()) {
              const auto vertex = face->vertex(v);
              const auto distance = vertex.norm();
              if (!(inner_radius - eps <= distance &&
                    distance <= outer_radius)) {
                face_on_annulus = false;
                break;
              }
            }
            if (face_on_annulus)
              face->set_all_manifold_ids(1);
          }
        }

        tria.set_manifold(1, SphericalManifold<dim>());
        dealii::TransfiniteInterpolationManifold<dim> transfinite_manifold;
        transfinite_manifold.initialize(tria);
        tria.set_manifold(0, transfinite_manifold);
      };

      /* Create inner ball with radius=inner_radius: */
      dealii::Triangulation<dim> tria_inner;
      {
        dealii::Triangulation<dim> temp;
        GridGenerator::hyper_ball_balanced(
            temp, dealii::Point<dim>(), inner_radius);
        temp.refine_global(2);
        GridGenerator::flatten_triangulation(temp, tria_inner);
      }

      /* Create outer annulus. Note part of this will be removed. */
      dealii::Triangulation<dim> annulus;
      GridGenerator::hyper_shell(
          annulus, dealii::Point<dim>(), inner_radius, outer_radius, 32);

      /* Create outside shell */
      dealii::Triangulation<dim> tria_outer;
      {
        dealii::Triangulation<dim> temp;
        GridGenerator::hyper_shell(temp,
                                   dealii::Point<dim>(),
                                   outer_radius,
                                   length / 2. * std::sqrt(2),
                                   8);
        /* Fix up vertices so that we get back a unit square: */
        for (const auto &cell : temp.cell_iterators()) {
          static_assert(dim == 2, "not implemented");
          for (unsigned int i = 0; i < 4; ++i) {
            auto &vertex = cell->vertex(i);
            if (std::abs(vertex[0]) < eps && std::abs(vertex[1]) > length / 2.)
              vertex[1] = std::copysign(length / 2., vertex[1]);
            if (std::abs(vertex[1]) < eps && std::abs(vertex[0]) > length / 2.)
              vertex[0] = std::copysign(length / 2., vertex[0]);
          }
        }
        assign_manifolds(temp);
        temp.refine_global(2);
        GridGenerator::flatten_triangulation(temp, tria_outer);
      }

      /* Create triangulation to merge: */
      dealii::Triangulation<dim, dim> coarse_triangulation;
      coarse_triangulation.set_mesh_smoothing(
          triangulation.get_mesh_smoothing());
      GridGenerator::merge_triangulations(
          {&tria_inner, &annulus, &tria_outer}, coarse_triangulation, 1.e-12);

      /*
       * Set manifold IDs:
       */

      coarse_triangulation.reset_all_manifolds();
      coarse_triangulation.set_all_manifold_ids(0);

      assign_manifolds(coarse_triangulation);
      coarse_triangulation.refine_global(2);

      /* Remove mesh cells in the annulus */

      std::set<typename dealii::Triangulation<dim>::active_cell_iterator>
          cells_to_remove;

      for (const auto &cell : coarse_triangulation.active_cell_iterators()) {
        for (auto f : dealii::GeometryInfo<dim>::face_indices()) {
          auto face = cell->face(f);
          const auto position = face->center();
          const auto radius = position.norm();
          const auto inner_value = inner_radius;
          const auto outer_value = outer_radius;

          bool in_anulus =
              radius - inner_value > 1.e-8 && outer_value - radius > 1.e-3;

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
          coarse_triangulation, cells_to_remove, coarse_triangulation);

      /*
       * Flatten triangulation and copy over to distributed triangulation:
       */
      dealii::Triangulation<dim> flattened_triangulation;
      flattened_triangulation.set_mesh_smoothing(
          triangulation.get_mesh_smoothing());
      GridGenerator::flatten_triangulation(coarse_triangulation,
                                           flattened_triangulation);
      triangulation.copy_triangulation(flattened_triangulation);
      assign_manifolds(triangulation);


      /*
       * Set boundary ids:
       */

      for (auto cell : triangulation.active_cell_iterators()) {
        for (auto f : GeometryInfo<dim>::face_indices()) {
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
    void annulus(Triangulation<3, 3> & /* triangulation */,
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
     * GridGenerator::annulus().
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
            "length", length_, "length of computational domain [-L/2,L/2]^d");

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
