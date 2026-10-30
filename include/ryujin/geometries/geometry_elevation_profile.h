//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/geometries/geometry_cut_cell_profile.h>

#include <deal.II/base/function_parser.h>

namespace ryujin
{
  namespace Geometries
  {
    /**
     * A modified rectangular domain, where the bottom boundary is
     * described by an elevation profile given as an expression in x. By
     * convention, the negative y-direction points to the bottom boundary.
     * See CutCellProfile for details.
     *
     * @note Only implemented in 2D.
     *
     * @ingroup Geometries
     */
    template <int dim>
    class ElevationProfile : public CutCellProfile<dim>
    {
    public:
      ElevationProfile(const std::string &subsection)
          : CutCellProfile<dim>("elevation profile", subsection)
      {
        elevation_expression_ = "0";
        this->add_parameter("elevation",
                            elevation_expression_,
                            "Elevation profile of the bottom boundary given "
                            "as an expression in x");
      }


      void create_coarse_triangulation(
          dealii::Triangulation<dim> &triangulation) const final
      {
        const dealii::FunctionParser<1> elevation(elevation_expression_);

        const auto height = [&](const dealii::Point<dim> &vertex) {
          return elevation.value(dealii::Point<1>(vertex[0]));
        };

        this->create_cut_cell_triangulation(triangulation, height);
      }

    private:
      std::string elevation_expression_;
    };

  } // namespace Geometries
} // namespace ryujin
