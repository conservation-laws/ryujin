//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2023 by the ryujin authors
//

#pragma once

#include "geometry_common_includes.h"

namespace ryujin
{
  namespace Geometries
  {
    /**
     * A simple rectangular domain that is used for most validation and
     * benchmark configurations.
     *
     * The rectangular domain is defined by two points, the bottom left
     * corner \f$(x_1,y_1,z_1)\f$ and the top right corner
     * \f$(x_2,y_2,z_2)\f$.
     *
     * A mesh grading can be enforced by defining an optional pull back and
     * push forward operation.
     *
     * By convenction the rectangular domain is orient with the x-axis to
     * from "left to right", the y-axis from "bottom to top" and the z-axis
     * from "the back towards the front".
     *
     * The class allows to prescribe any of the supported boundary
     * condition on any of the 2, 4, or 6 faces.
     *
     * @ingroup Mesh
     */
    template <int dim>
    class RectangularDomain : public Geometry<dim>
    {
    public:
      RectangularDomain(const std::string subsection)
          : Geometry<dim>("rectangular domain", subsection)
      {
        this->add_parameter("position bottom left",
                            point_left_,
                            "Position of bottom left corner");

        for (unsigned int d = 0; d < dim; ++d)
          point_right_[d] = 20.0;
        this->add_parameter(
            "position top right", point_right_, "Position of top right corner");

        grading_push_forward_ = dim == 1 ? "x" : (dim == 2 ? "x;y" : "x;y;z;");
        this->add_parameter("grading push forward",
                            grading_push_forward_,
                            "push forward of grading manifold");

        grading_pull_back_ = grading_push_forward_;
        this->add_parameter("grading pull back",
                            grading_pull_back_,
                            "pull back of grading manifold");

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

        /* create grading: */

        // FIXME: We should ideally check the push forward and pull back
        // for compatiblity.
        if (grading_push_forward_ != grading_pull_back_) {
          dealii::FunctionManifold<dim> grading(grading_push_forward_,
                                                grading_pull_back_);
          triangulation.set_all_manifold_ids(1);
          triangulation.set_manifold(1, grading);
        }

        /* set boundary ids: */

        for (auto cell : triangulation.active_cell_iterators()) {
          for (auto f : dealii::GeometryInfo<dim>::face_indices()) {
            auto face = cell->face(f);
            if (!face->at_boundary())
              continue;
            const auto position = face->center();

            if (position[0] < point_left_[0] + 1.e-8)
              face->set_boundary_id(boundary_left_);
            if (position[0] > point_right_[0] - 1.e-8)
              face->set_boundary_id(boundary_right_);

            if constexpr (dim >= 2) {
              if (position[1] < point_left_[1] + 1.e-8)
                face->set_boundary_id(boundary_bottom_);
              if (position[1] > point_right_[1] - 1.e-8)
                face->set_boundary_id(boundary_top_);
            }

            if constexpr (dim == 3) {
              if (position[2] < point_left_[2] + 1.e-8)
                face->set_boundary_id(boundary_back_);
              if (position[2] > point_right_[2] - 1.e-8)
                face->set_boundary_id(boundary_front_);
            }
          } /*for*/
        } /*for*/

        std::vector<int> directions;

        if (boundary_left_ == Boundary::periodic ||
            boundary_right_ == Boundary::periodic) {
          AssertThrow(boundary_left_ == boundary_right_,
                      dealii::ExcMessage(
                          "For prescribing periodic boundaries in x-direction, "
                          "both, the left and right boundary conditions must "
                          "be set to periodic"));
          directions.push_back(0);
        }

        if (dim >= 2 && (boundary_bottom_ == Boundary::periodic ||
                         boundary_top_ == Boundary::periodic)) {
          AssertThrow(boundary_bottom_ == boundary_top_,
                      dealii::ExcMessage(
                          "For prescribing periodic boundaries in y-direction, "
                          "both, the bottom and top boundary conditions must "
                          "be set to periodic"));
          directions.push_back(1);
        }

        if (dim == 3 && (boundary_back_ == Boundary::periodic ||
                         boundary_front_ == Boundary::periodic)) {
          AssertThrow(boundary_back_ == boundary_front_,
                      dealii::ExcMessage(
                          "For prescribing periodic boundaries in z-direction, "
                          "both, the back and front boundary conditions must "
                          "be set to periodic"));
          directions.push_back(2);
        }

#if DEAL_II_VERSION_GTE(9, 5, 0)
        if (!directions.empty()) {
          std::vector<dealii::GridTools::PeriodicFacePair<
              typename dealii::Triangulation<dim>::cell_iterator>>
              periodic_faces;

          for (const auto direction : directions)
            dealii::GridTools::collect_periodic_faces(
                triangulation,
                /*b_id */ Boundary::periodic,
                /*direction*/ direction,
                periodic_faces);

          triangulation.add_periodicity(periodic_faces);
        }
#endif
      }

    private:
      dealii::Point<dim> point_left_;
      dealii::Point<dim> point_right_;

      unsigned int subdivisions_x_;
      unsigned int subdivisions_y_;
      unsigned int subdivisions_z_;

      std::string grading_push_forward_;
      std::string grading_pull_back_;

      Boundary boundary_back_;
      Boundary boundary_bottom_;
      Boundary boundary_front_;
      Boundary boundary_left_;
      Boundary boundary_right_;
      Boundary boundary_top_;
    };
  } /* namespace Geometries */
} /* namespace ryujin */
