//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "geometry_common_includes.h"

namespace ryujin
{
  namespace GridGenerator
  {
    /**
     * Create a 2D numerical tank configuration.
     *
     * The domain consists of two "tanks" attached via a "tunnel".
     * For the sake of simplicity, we assume a couple of things:
     *
     *    1. the two tanks are the same size,
     *    2. the centerline of tanks and tunnel align at y = 0.
     *
     * The walls of the tunnel have "slip" (or wall / reflecting) boundary
     * conditions. The "inner" walls also have slip boundary conditions.
     * For the left/right external boundaries, we assume "dynamic".
     *
     * NOTE: The default parameters are arbitrary.
     *
     * @ingroup Mesh
     */

    template <int dim, int spacedim, template <int, int> class Triangulation>
    void twotanks(Triangulation<dim, spacedim> &,
                  const double /*tank_length*/,
                  const double /*tank_width*/,
                  const double /*tunnel_length*/,
                  const double /*tunnel_width*/,
                  const unsigned int /*subdivisions_factor*/)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
      __builtin_trap();
    }


#ifndef DOXYGEN
    template <template <int, int> class Triangulation>
    void twotanks(Triangulation<2, 2> &triangulation,
                  const double tank_length,
                  const double tank_width,
                  const double tunnel_length,
                  const double tunnel_width,
                  const unsigned int subdivisions_factor)
    {
      using namespace dealii;

      dealii::Triangulation<2, 2> res1, res2, res3, tank1, tank2, tunnel, final;

      const double tolerance = 1.e-8;

      Assert(
          tank_width - tunnel_width > tolerance,
          dealii::ExcMessage(
              " !!! The tank width must be larger than the tunnel width !!!"));

      /* We split the tank into three triangulations and subdivide to
       * get somewhat close to uniform refinement */

      const double diff = (tank_width - tunnel_width) / 2.;
      unsigned int sub_x =
          static_cast<int>(std::round(tank_length * subdivisions_factor));
      unsigned int sub_y =
          static_cast<int>(std::round(diff * subdivisions_factor));

      GridGenerator::subdivided_hyper_rectangle(
          res1,
          {sub_x, sub_y},
          Point<2>(-tank_length, -tank_width / 2.),
          Point<2>(0, -tunnel_width / 2.));

      GridGenerator::subdivided_hyper_rectangle(
          res3,
          {sub_x, sub_y},
          Point<2>(-tank_length, tunnel_width / 2.),
          Point<2>(0, tank_width / 2.));

      sub_y = static_cast<int>(std::round(tunnel_width * subdivisions_factor));

      GridGenerator::subdivided_hyper_rectangle(
          res2,
          {sub_x, sub_y},
          Point<2>(-tank_length, -tunnel_width / 2.),
          Point<2>(0, tunnel_width / 2.));

      // We create tank1 by merging the three triangulations above
      tank1.set_mesh_smoothing(triangulation.get_mesh_smoothing());
      GridGenerator::merge_triangulations(
          {&res1, &res2, &res3}, tank1, tolerance);

      // We now create the second tank (tank2) by copying the above and shifting
      tank2.copy_triangulation(tank1);
      dealii::Point<2> shift_vector(tunnel_length + tank_length, 0.);
      dealii::GridTools::shift(shift_vector, tank2);

      // We now create the tunnel
      sub_x = static_cast<int>(std::round(tunnel_length * subdivisions_factor));

      GridGenerator::subdivided_hyper_rectangle(
          tunnel,
          {sub_x, sub_y},
          Point<2>(0., -tunnel_width / 2.),
          Point<2>(tunnel_length, tunnel_width / 2.));


      // We now merge the two tanks and the tunnel
      final.set_mesh_smoothing(triangulation.get_mesh_smoothing());
      GridGenerator::merge_triangulations(
          {&tank1, &tunnel, &tank2}, final, tolerance);


      // Finally, copy the "final" triangulation to "triangulation"
      triangulation.copy_triangulation(final);

      /*
       * Set boundary ids:
       */

      for (auto cell : triangulation.active_cell_iterators()) {
        for (auto f : cell->face_indices()) {
          const auto face = cell->face(f);

          if (!face->at_boundary())
            continue;

          /*
           * We want slip everywhere except the left/right edges of tanks.
           */

          face->set_boundary_id(Boundary::slip);

          const auto center = face->center();
          if (center[0] > tank_length + tunnel_length - tolerance)
            face->set_boundary_id(Boundary::dynamic);

          if (center[0] < -tank_length + tolerance)
            face->set_boundary_id(Boundary::dynamic);

        } /*f*/
      }   /*cell*/
    }
#endif
  } /* namespace GridGenerator */


  namespace Geometries
  {
    /**
     * A 2D tank configuration constructed with GridGenerator::twotanks().
     *
     * @ingroup Mesh
     */
    template <int dim>
    class TwoTanks : public Geometry<dim>
    {
    public:
      TwoTanks(const std::string &subsection)
          : Geometry<dim>("two tanks", subsection)
      {
        tank_length_ = 100.;
        this->add_parameter(
            "tank length", tank_length_, "length of tanks [units]");

        tank_width_ = 100.;
        this->add_parameter("tank width", tank_width_, "width of tank [units]");

        tunnel_length_ = 10.;
        this->add_parameter(
            "tunnel length", tunnel_length_, "length of tunnel [units]");

        tunnel_width_ = 50.;
        this->add_parameter(
            "tunnel width", tunnel_width_, "width of tunnel [units]");

        // if you divide the default values by 100, make this number 100
        subdivisions_factor_ = 1;
        this->add_parameter("subdivisions factor",
                            subdivisions_factor_,
                            "A number used for introducing subdivions in both "
                            "x-y direction. Useful when dealing with "
                            "measurements that are less than 1. ");
      }

      void create_coarse_triangulation(
          dealii::Triangulation<dim> &triangulation) const final
      {
        GridGenerator::twotanks(triangulation,
                                tank_length_,
                                tank_width_,
                                tunnel_length_,
                                tunnel_width_,
                                subdivisions_factor_);
      }

    private:
      double tank_length_;
      double tank_width_;
      double tunnel_length_;
      double tunnel_width_;

      unsigned int subdivisions_factor_;
    };
  } /* namespace Geometries */
} /* namespace ryujin */
