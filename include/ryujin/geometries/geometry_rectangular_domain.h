//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/geometries/geometry_common_includes.h>

#include <deal.II/base/function_parser.h>

namespace ryujin
{
  namespace Geometries
  {
    /**
     * A simple rectangular domain that is used for most validation and
     * benchmark configurations.
     *
     * For @p structdim < @p dim the rectangle is embedded into the plane
     * spanned by the first @p structdim coordinate axes. The corners are
     * given in these intrinsic coordinates; an optional transformation
     * (given as dim expressions in the coordinates of the embedding space)
     * deforms the rectangle into a curved manifold.
     *
     * The rectangular domain is defined by two points, the bottom left
     * corner \f$(x_1,y_1,z_1)\f$ and the top right corner
     * \f$(x_2,y_2,z_2)\f$.
     *
     * A mesh grading can be enforced by defining an optional pull back and
     * push forward operation.
     *
     * Alternatively, the domain can be deformed by an optional forward
     * transformation given as a semicolon separated list of dim
     * expressions in x, y, z. If set, the geometry is realized with a
     * MappingQCache, see Geometry::transformation().
     *
     * By convention the rectangular domain is orient with the x-axis to
     * from "left to right", the y-axis from "bottom to top" and the z-axis
     * from "the back towards the front".
     *
     * The class allows to prescribe any of the supported boundary
     * condition on any of the 2, 4, or 6 faces.
     *
     * @ingroup Geometries
     */
    template <int structdim, int dim = structdim>
    class RectangularDomain : public Geometry<structdim, dim>
    {
    public:
      RectangularDomain(const std::string &name, const std::string &subsection)
          : Geometry<structdim, dim>(name, subsection)
      {
        this->add_parameter("position bottom left",
                            point_left_,
                            "Position of bottom left corner");

        for (unsigned int d = 0; d < structdim; ++d)
          point_right_[d] = 20.0;
        this->add_parameter(
            "position top right", point_right_, "Position of top right corner");

        grading_push_forward_ = dim == 1 ? "x" : (dim == 2 ? "x;y" : "x;y;z;");
        grading_pull_back_ = grading_push_forward_;
        if constexpr (structdim == dim) {
          this->add_parameter("grading push forward",
                              grading_push_forward_,
                              "push forward of grading manifold");
          this->add_parameter("grading pull back",
                              grading_pull_back_,
                              "pull back of grading manifold");
        }

        this->add_parameter(
            "transformation",
            transformation_expression_,
            "Optional forward transformation of the rectangular domain given "
            "as a semicolon separated list of dim expressions in x, y, z. If "
            "empty, no transformation is applied.");

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

        if constexpr (structdim >= 2) {
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

        if constexpr (structdim == 3) {
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

        use_simplices_ = false;
        this->add_parameter("simplex mesh",
                            use_simplices_,
                            "If set to true, the triangulation uses simplices "
                            "instead of quadrangles.");
      }


      RectangularDomain(const std::string subsection)
          : RectangularDomain("rectangular domain", subsection)
      {
      }


      void create_coarse_triangulation(
          dealii::Triangulation<structdim, dim> &triangulation) const override
      {
        using dealii::GridGenerator::subdivided_hyper_rectangle_with_simplices;

        /* create mesh: */

        dealii::Triangulation<structdim> tria1;
        if constexpr (structdim == dim)

          tria1.set_mesh_smoothing(triangulation.get_mesh_smoothing());

        if (use_simplices_) {
          /* Mesh with simplices: */

          if constexpr (structdim == 1) {
            subdivided_hyper_rectangle_with_simplices<1, 1>(
                tria1, {subdivisions_x_}, point_left_, point_right_);
          } else if constexpr (structdim == 2) {
            subdivided_hyper_rectangle_with_simplices(
                tria1,
                {subdivisions_x_, subdivisions_y_},
                point_left_,
                point_right_);
          } else if constexpr (structdim == 3) {
            subdivided_hyper_rectangle_with_simplices(
                tria1,
                {subdivisions_x_, subdivisions_y_, subdivisions_z_},
                point_left_,
                point_right_);
          }

        } else {
          /* Regular mesh with quadrilaterals/hexahedra: */

          if constexpr (structdim == 1) {
            dealii::GridGenerator::subdivided_hyper_rectangle<1, 1>(
                tria1, {subdivisions_x_}, point_left_, point_right_);
          } else if constexpr (structdim == 2) {
            dealii::GridGenerator::subdivided_hyper_rectangle(
                tria1,
                {subdivisions_x_, subdivisions_y_},
                point_left_,
                point_right_);
          } else if constexpr (structdim == 3) {
            dealii::GridGenerator::subdivided_hyper_rectangle(
                tria1,
                {subdivisions_x_, subdivisions_y_, subdivisions_z_},
                point_left_,
                point_right_);
          }
        }

        /* Set boundary ids and periodicity on a given triangulation: */

        const auto set_boundary_ids = [this](auto &tria) {
          for (auto cell : tria.active_cell_iterators()) {
            for (auto f : cell->face_indices()) {
              auto face = cell->face(f);
              if (!face->at_boundary())
                continue;
              const auto position = face->center();

              /*
               * The local part of a fully distributed mesh has spurious
               * boundary faces that do not lie on the boundary of the global
               * domain. Do not let these inherit a default boundary id:
               */
              face->set_boundary_id(Boundary::do_nothing);

              if (position[0] < point_left_[0] + 1.e-8)
                face->set_boundary_id(boundary_left_);
              if (position[0] > point_right_[0] - 1.e-8)
                face->set_boundary_id(boundary_right_);

              if constexpr (structdim >= 2) {
                if (position[1] < point_left_[1] + 1.e-8)
                  face->set_boundary_id(boundary_bottom_);
                if (position[1] > point_right_[1] - 1.e-8)
                  face->set_boundary_id(boundary_top_);
              }

              if constexpr (structdim == 3) {
                if (position[2] < point_left_[2] + 1.e-8)
                  face->set_boundary_id(boundary_back_);
                if (position[2] > point_right_[2] - 1.e-8)
                  face->set_boundary_id(boundary_front_);
              }
            } /*for*/
          }   /*for*/
        };

        const auto add_periodicity = [this](auto &tria [[maybe_unused]]) {
          std::vector<int> directions;

          if (boundary_left_ == Boundary::periodic ||
              boundary_right_ == Boundary::periodic) {
            AssertThrow(
                boundary_left_ == boundary_right_,
                dealii::ExcMessage(
                    "For prescribing periodic boundaries in x-direction, "
                    "both, the left and right boundary conditions must "
                    "be set to periodic"));
            directions.push_back(0);
          }

          if (structdim >= 2 && (boundary_bottom_ == Boundary::periodic ||
                                 boundary_top_ == Boundary::periodic)) {
            AssertThrow(
                boundary_bottom_ == boundary_top_,
                dealii::ExcMessage(
                    "For prescribing periodic boundaries in y-direction, "
                    "both, the bottom and top boundary conditions must "
                    "be set to periodic"));
            directions.push_back(1);
          }

          if (structdim == 3 && (boundary_back_ == Boundary::periodic ||
                                 boundary_front_ == Boundary::periodic)) {
            AssertThrow(
                boundary_back_ == boundary_front_,
                dealii::ExcMessage(
                    "For prescribing periodic boundaries in z-direction, "
                    "both, the back and front boundary conditions must "
                    "be set to periodic"));
            directions.push_back(2);
          }

#ifndef BUG_COLLECT_PERIODIC_FACES_INSTANTIATION
          if (!directions.empty()) {
            std::vector<dealii::GridTools::PeriodicFacePair<
                typename std::remove_cvref_t<decltype(tria)>::cell_iterator>>
                periodic_faces;

            for (const auto direction : directions)
              dealii::GridTools::collect_periodic_faces(
                  tria,
                  /*b_id */ Boundary::periodic,
                  /*direction*/ direction,
                  periodic_faces);

            tria.add_periodicity(periodic_faces);
          }
#endif
        };

        if constexpr (structdim == dim) {
          /*
           * A parallel::fullydistributed::Triangulation only stores locally
           * relevant cells and determines them when copying, so the source
           * mesh must already know about periodic neighbors:
           */
          using FD = dealii::parallel::fullydistributed::Triangulation<dim>;
          if (dynamic_cast<FD *>(&triangulation) != nullptr) {
            set_boundary_ids(tria1);
            add_periodicity(tria1);
          }
          triangulation.copy_triangulation(tria1);
        } else
          dealii::GridGenerator::flatten_triangulation(tria1, triangulation);

        /* create transformation: */

        if (!transformation_expression_.empty()) {
          AssertThrow(!use_simplices_,
                      dealii::ExcMessage(
                          "A transformation can only be used with a mesh "
                          "consisting of quadrilaterals or hexahedra."));

          const auto transformation =
              std::make_shared<dealii::FunctionParser<dim>>(
                  transformation_expression_);
          AssertThrow(transformation->n_components == dim,
                      dealii::ExcMessage(
                          "The transformation must consist of exactly dim "
                          "semicolon separated expressions."));

          this->transformation_ =
              [transformation](
                  const typename dealii::Triangulation<structdim,
                                                       dim>::cell_iterator &,
                  const dealii::Point<dim> &point) {
                dealii::Point<dim> result;
                for (unsigned int d = 0; d < dim; ++d)
                  result[d] = transformation->value(point, d);
                return result;
              };
        }

        /* create grading: */

        // FIXME: We should ideally check the push forward and pull back
        // for compatiblity.
        if constexpr (structdim == dim) {
          if (grading_push_forward_ != grading_pull_back_) {
            dealii::FunctionManifold<dim> grading(grading_push_forward_,
                                                  grading_pull_back_);
            triangulation.set_all_manifold_ids(1);
            triangulation.set_manifold(1, grading);
          }
        }

        set_boundary_ids(triangulation);
        add_periodicity(triangulation);
      }


      using HP_Collection = typename Geometry<structdim, dim>::HP_Collection;
      using Collection = typename Discretization<structdim, dim>::Collection;

      HP_Collection
      populate_hp_collections(const unsigned int /*fe_degree*/,
                              Collection & /*collection*/) const override
      {
        if (use_simplices_) {
          return HP_Collection::standard_simplices;
        } else {
          return HP_Collection::standard_quadrilaterals;
        }
      }


    protected:
      dealii::Point<structdim> point_left_;
      dealii::Point<structdim> point_right_;

      unsigned int subdivisions_x_;
      unsigned int subdivisions_y_;
      unsigned int subdivisions_z_;

      std::string grading_push_forward_;
      std::string grading_pull_back_;

      std::string transformation_expression_;

      Boundary boundary_back_;
      Boundary boundary_bottom_;
      Boundary boundary_front_;
      Boundary boundary_left_;
      Boundary boundary_right_;
      Boundary boundary_top_;

      bool use_simplices_;
    };
  } // namespace Geometries
} // namespace ryujin
