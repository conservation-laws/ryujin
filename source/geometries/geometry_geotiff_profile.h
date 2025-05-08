//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2023 by the ryujin authors
//

#pragma once

#include "geometry_common_includes.h"
#include "geotiff_reader.h"

namespace ryujin
{
  namespace Geometries
  {
    /**
     * A ChartManifold that warps the y-direction (in 2D) or z-direction
     * (in 3D) with a given function callable. The callable lambda must
     * take a dealii::Point<dim> as argument and return a double that is
     * used for the shift. The computation of the shift must only depend on
     * the x-coordinate (in 2D) or the x and y coordinates (in 3D).
     */
    template <int dim, typename Callable>
    class ProfileManifold : public dealii::ChartManifold<dim>
    {
    public:
      ProfileManifold(const Callable &callable)
          : callable_(callable)
      {
      }

      dealii::Point<dim>
      pull_back(const dealii::Point<dim> &space_point) const final
      {
        auto chart_point = space_point;

        if constexpr (dim >= 2) {
          /* transform y-direction (2D) or z-direction (3D): */
          chart_point[dim - 1] -= callable_(space_point);
        }

        return chart_point;
      }

      dealii::Point<dim>
      push_forward(const dealii::Point<dim> &chart_point) const final
      {
        auto space_point = chart_point;

        if constexpr (dim >= 2) {
          /* transform y-direction (2D) or z-direction (3D): */
          space_point[dim - 1] += callable_(space_point);
        }

        return space_point;
      }

      std::unique_ptr<dealii::Manifold<dim, dim>> clone() const final
      {
        return std::make_unique<ProfileManifold<dim, Callable>>(callable_);
      }

    private:
      const Callable callable_;
    };


    template <int dim, typename Callable>
    ProfileManifold<dim, Callable>
    make_profile_manifold(const Callable &callable)
    {
      return {callable};
    }


    /**
     * @ingroup Mesh
     */
    template <int dim>
    class GeoTIFFProfile : public Geometry<dim>
    {
    public:
      GeoTIFFProfile(const std::string subsection)
          : Geometry<dim>("geotiff profile", subsection)
          , geotiff_reader_(subsection + "/geotiff profile")
      {
        this->add_parameter("position bottom left",
                            point_left_,
                            "Position of bottom left corner");

        for (unsigned int d = 0; d < dim; ++d)
          point_right_[d] = 20.0;
        this->add_parameter(
            "position top right", point_right_, "Position of top right corner");

        subdivisions_x_ = 1;
        subdivisions_y_ = 1;
        subdivisions_z_ = 1;
        boundary_back_ = Boundary::dirichlet;
        boundary_bottom_ = Boundary::dirichlet;
        boundary_front_ = Boundary::dirichlet;
        boundary_left_ = Boundary::dirichlet;
        boundary_right_ = Boundary::dirichlet;
        boundary_top_ = Boundary::dirichlet;

        this->add_parameter("subdivisions x",
                            subdivisions_x_,
                            "number of subdivisions in x direction");
        this->add_parameter(
            "boundary condition left",
            boundary_left_,
            "Type of boundary condition enforced on the left side of the "
            "domain (faces with normal in negative x direction)");
        this->add_parameter(
            "boundary condition right",
            boundary_right_,
            "Type of boundary condition enforced on the right side of the "
            "domain (faces with normal in positive x direction)");

        if constexpr (dim >= 2) {
          this->add_parameter("subdivisions y",
                              subdivisions_y_,
                              "number of subdivisions in y direction");
          this->add_parameter(
              "boundary condition bottom",
              boundary_bottom_,
              "Type of boundary condition enforced on the bottom side of the "
              "domain (faces with normal in negative y direction)");
          this->add_parameter(
              "boundary condition top",
              boundary_top_,
              "Type of boundary condition enforced on the top side of the "
              "domain (faces with normal in positive y direction)");
        }

        if constexpr (dim == 2) {
          reference_y_coordinate_ = 0.;
          this->add_parameter(
              "reference y coordinate",
              reference_y_coordinate_,
              "GeoTIFF: select the value for y-coordinate in 2D. That is, the "
              "1D profile for the lower boundary is queried from the 2D "
              "geotiff image at coordinates (x, y=constant)");
        }

        if constexpr (dim == 3) {
          this->add_parameter("subdivisions z",
                              subdivisions_z_,
                              "number of subdivisions in z direction");
          this->add_parameter(
              "boundary condition back",
              boundary_back_,
              "Type of boundary condition enforced on the back side of the "
              "domain (faces with normal in negative z direction)");
          this->add_parameter(
              "boundary condition front",
              boundary_front_,
              "Type of boundary condition enforced on the front side of the "
              "domain (faces with normal in positive z direction)");
        }
      }


      void create_triangulation(
          typename Geometry<dim>::Triangulation &triangulation) final
      {
        /* create mesh: */

        dealii::Triangulation<dim, dim> tria1;
        tria1.set_mesh_smoothing(triangulation.get_mesh_smoothing());

        if constexpr (dim == 1) {
          dealii::GridGenerator::subdivided_hyper_rectangle<dim, dim>(
              tria1, {subdivisions_x_}, point_left_, point_right_);
        } else if constexpr (dim == 2) {
          dealii::GridGenerator::subdivided_hyper_rectangle(
              tria1,
              {subdivisions_x_, subdivisions_y_},
              point_left_,
              point_right_);
        } else if constexpr (dim == 3) {
          dealii::GridGenerator::subdivided_hyper_rectangle(
              tria1,
              {subdivisions_x_, subdivisions_y_, subdivisions_z_},
              point_left_,
              point_right_);
        }

        triangulation.copy_triangulation(tria1);
        triangulation.reset_all_manifolds();
        /* manifold id 0 for transfinite interpolation manifold */
        triangulation.set_all_manifold_ids(0);

        /* set boundary and manifold ids: */

        for (auto cell : triangulation.active_cell_iterators()) {
          for (auto f : cell->face_indices()) {
            auto face = cell->face(f);
            if (!face->at_boundary())
              continue;
            const auto position = face->center();

            if (position[0] < point_left_[0] + 1.e-8) {
              face->set_boundary_id(boundary_left_);
              face->set_manifold_id(dealii::numbers::flat_manifold_id);
            }

            if (position[0] > point_right_[0] - 1.e-8) {
              face->set_boundary_id(boundary_right_);
              face->set_manifold_id(dealii::numbers::flat_manifold_id);
            }


            if constexpr (dim == 2) {
              if (position[1] < point_left_[1] + 1.e-8) {
                face->set_boundary_id(boundary_bottom_);
                /* manifold id 1 for ProfileManifold: */
                face->set_manifold_id(1);
              }
              if (position[1] > point_right_[1] - 1.e-8) {
                face->set_boundary_id(boundary_top_);
                face->set_manifold_id(dealii::numbers::flat_manifold_id);
              }
            }

            if constexpr (dim == 3) {
              if (position[1] < point_left_[1] + 1.e-8) {
                face->set_boundary_id(boundary_bottom_);
                face->set_manifold_id(dealii::numbers::flat_manifold_id);
              }

              if (position[1] > point_right_[1] - 1.e-8) {
                face->set_boundary_id(boundary_top_);
                face->set_manifold_id(dealii::numbers::flat_manifold_id);
              }

              /*
               * The lower boundary at z = point_left_[2] is the profile
               * manifold ant the upper boundary boundary at z =
               * point_right_[2] is flat:
               */

              if (position[2] < point_left_[2] + 1.e-8) {
                face->set_boundary_id(boundary_back_);
                /* manifold id 1 for ProfileManifold: */
                face->set_manifold_id(1);
              }

              if (position[2] > point_right_[2] - 1.e-8) {
                face->set_boundary_id(boundary_front_);
                face->set_manifold_id(dealii::numbers::flat_manifold_id);
              }
            }
          } /*for*/
        }   /*for*/

        const auto profile =
            make_profile_manifold<dim>([&](dealii::Point<dim> point) {
              /*
               *
               */
              if constexpr (dim == 1) {
                return 0.;
              } else if constexpr (dim == 2) {
                /*
                 * Set the second coordinate to a constant when querying
                 * height information in 2D.
                 */
                point[1] = reference_y_coordinate_;
                return geotiff_reader_.compute_height(point);
              } else if constexpr (dim == 3) {
                return geotiff_reader_.compute_height(point);
              }
            });
        triangulation.set_manifold(1, profile);

        dealii::TransfiniteInterpolationManifold<dim> transfinite_interpolation;
        transfinite_interpolation.initialize(triangulation);
        triangulation.set_manifold(0, transfinite_interpolation);
      }

    private:
      GeoTIFFReader<dim> geotiff_reader_;

      dealii::Point<dim> point_left_;
      dealii::Point<dim> point_right_;

      double reference_y_coordinate_;

      unsigned int subdivisions_x_;
      unsigned int subdivisions_y_;
      unsigned int subdivisions_z_;

      Boundary boundary_back_;
      Boundary boundary_bottom_;
      Boundary boundary_front_;
      Boundary boundary_left_;
      Boundary boundary_right_;
      Boundary boundary_top_;
    };
  } /* namespace Geometries */
} /* namespace ryujin */
