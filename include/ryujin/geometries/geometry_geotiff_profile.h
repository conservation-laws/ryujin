//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/geotiff_reader.h>
#include <ryujin/geometries/geometry_cut_cell_profile.h>

namespace ryujin
{
  namespace Geometries
  {
    /**
     * A modified rectangular domain, where the bottom boundary is
     * described by an elevation profile read from a GeoTIFF file. By
     * convention, the negative y-direction points to the bottom boundary.
     * See CutCellProfile for details.
     *
     * @note Only implemented in 2D.
     *
     * @ingroup Geometries
     */
    template <int dim>
    class GeoTIFFProfile : public CutCellProfile<dim>
    {
    public:
      GeoTIFFProfile(const std::string &subsection)
          : CutCellProfile<dim>("geotiff profile", subsection)
          , geotiff_reader_(subsection + "/geotiff profile/geotiff")
      {
        reference_y_coordinate_ = 0.;
        this->add_parameter(
            "reference y coordinate",
            reference_y_coordinate_,
            "GeoTIFF: select the value for y-coordinate in 2D. That is, the "
            "1D profile for the lower boundary is queried from the 2D "
            "geotiff image at coordinates (x, y=constant)");
      }


      void create_coarse_triangulation(
          dealii::Triangulation<dim> &triangulation) const final
      {
        /*
         * The 1D profile is extracted from the geotiff image along the
         * line y = reference_y_coordinate_:
         */
        const auto height = [&](const dealii::Point<dim> &vertex) {
          return geotiff_reader_.compute_height(
              dealii::Point<2>(vertex[0], reference_y_coordinate_));
        };

        this->create_cut_cell_triangulation(triangulation, height);
      }

    private:
      GeoTIFFReader geotiff_reader_;

      double reference_y_coordinate_;
    };

  } // namespace Geometries
} // namespace ryujin
