//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// [LANL Copyright Statement]
// Copyright (C) 2024 by the ryujin authors
// Copyright (C) 2024 by Triad National Security, LLC
//

#pragma once

#include "geometry_common_includes.h"

namespace ryujin
{
  namespace Geometries
  {
    /**
     * A 2D disk / 3D ball configuration constructed with
     * GridGenerator::hyper_ball_balanced().
     *
     * @ingroup Mesh
     */
    template <int dim>
    class Disk : public Geometry<dim>
    {
    public:
      Disk(const std::string subsection)
          : Geometry<dim>("disk", subsection)
      {
        radius_ = 1.2;
        this->add_parameter("radius", radius_, "radius of disk");

        boundary_ = Boundary::dirichlet;
        this->add_parameter("boundary condition",
                            boundary_,
                            "Type of boundary condition enforced on the "
                            "boundary of the disk/ball");
      }

      void create_triangulation(
          typename Geometry<dim>::Triangulation &triangulation) final
      {
        GridGenerator::hyper_ball_balanced(
            triangulation, dealii::Point<dim>(), radius_);

        /*
         * Set boundary ids:
         */
        for (auto cell : triangulation.active_cell_iterators()) {
          for (auto f : cell->face_indices()) {
            const auto face = cell->face(f);

            if (!face->at_boundary())
              continue;

            face->set_boundary_id(boundary_);
          }
        }
      }

    private:
      double radius_;
      Boundary boundary_;
    };
  } /* namespace Geometries */
} /* namespace ryujin */
