//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include "geometry_common_includes.h"

namespace ryujin
{
  namespace GridGenerator
  {
    /**
     * Create a 2D numerical wave tank with specifications outlined in Figure 4
     * of @cite 2018-SW-exps. The tank consists of a rectangular water reservoir
     * with a long flume attached to the "center" of the reservoir. For the sake
     * of simplicity, we assume the centerline of the flume and reservoir align
     * at y = 0.
     *
     * All outer walls have "slip" boundary conditions except outer wall at the
     * end of the tank which has dynamic outflow conditions.
     *
     * @ingroup Mesh
     */

    template <int dim, int spacedim, template <int, int> class Triangulation>
    void wavetank(Triangulation<dim, spacedim> &,
                  const double /*reservoir_length*/,
                  const double /*reservoir_width*/,
                  const double /*flume_length*/,
                  const double /*flume_width*/)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
      __builtin_trap();
    }


#ifndef DOXYGEN
    template <template <int, int> class Triangulation>
    void wavetank(Triangulation<2, 2> &triangulation,
                  const double reservoir_length,
                  const double reservoir_width,
                  const double flume_length,
                  const double flume_width)
    {
      using namespace dealii;

      dealii::Triangulation<2, 2> res1, res2, res3, flume, final;

      const double tolerance = 1.e-8;

      Assert(reservoir_width - flume_width > tolerance,
             dealii::ExcInternalError());

      /* We split the reservoir into three triangulations and subdivide to
       * get close to uniform refinement */

      const double diff = (reservoir_width - flume_width) / 2.;
      unsigned int sub_x = int(std::round(reservoir_length * 100.));
      unsigned int sub_y = int(std::round(diff * 100.));

      GridGenerator::subdivided_hyper_rectangle(
          res1,
          {sub_x, sub_y},
          Point<2>(-reservoir_length, -reservoir_width / 2.),
          Point<2>(0, -flume_width / 2.));

      GridGenerator::subdivided_hyper_rectangle(
          res3,
          {sub_x, sub_y},
          Point<2>(-reservoir_length, flume_width / 2.),
          Point<2>(0, reservoir_width / 2.));

      sub_y = int(std::round(flume_width * 100.));

      GridGenerator::subdivided_hyper_rectangle(
          res2,
          {sub_x, sub_y},
          Point<2>(-reservoir_length, -flume_width / 2.),
          Point<2>(0, flume_width / 2.));

      sub_x = int(std::round(flume_length * 100.));

      GridGenerator::subdivided_hyper_rectangle(
          flume,
          {sub_x, sub_y},
          Point<2>(0., -flume_width / 2.),
          Point<2>(flume_length, flume_width / 2.));

      final.set_mesh_smoothing(triangulation.get_mesh_smoothing());
      GridGenerator::merge_triangulations(
          {&res1, &res2, &res3, &flume}, final, tolerance);
      triangulation.copy_triangulation(final);

      /*
       * Set boundary ids:
       */

      for (auto cell : triangulation.active_cell_iterators()) {
        for (auto f : GeometryInfo<2>::face_indices()) {
          const auto face = cell->face(f);

          if (!face->at_boundary())
            continue;

          /*
           * We want slip everywhere except right edge of tank.
           */

          face->set_boundary_id(Boundary::slip);

          const auto center = face->center();
          if (center[0] > flume_length - tolerance)
            face->set_boundary_id(Boundary::dynamic);

        } /*f*/
      }   /*cell*/
    }
#endif
  } /* namespace GridGenerator */


  namespace Geometries
  {
    /**
     * A 2D wave tank configuration constructed with GridGenerator::wavetank().
     *
     * @ingroup Mesh
     */
    template <int dim>
    class WaveTank : public Geometry<dim>
    {
    public:
      WaveTank(const std::string subsection)
          : Geometry<dim>("wave tank", subsection)
      {
        reservoir_length_ = 157. / 100.;
        this->add_parameter("reservoir length",
                            reservoir_length_,
                            "length of water reservoir [meters]");

        reservoir_width_ = 8.1 / 100.;
        this->add_parameter("reservoir width",
                            reservoir_width_,
                            "width of water reservoir [meters]");

        flume_length_ = 600.78 / 100.;
        this->add_parameter(
            "flume length", flume_length_, "length of flume [meters]");

        flume_width_ = 24 / 100.;
        this->add_parameter(
            "flume width", flume_width_, "width of flume [meters]");
      }

      void create_triangulation(
          typename Geometry<dim>::Triangulation &triangulation) final
      {
        GridGenerator::wavetank(triangulation,
                                reservoir_length_,
                                reservoir_width_,
                                flume_length_,
                                flume_width_);
      }

    private:
      double reservoir_length_;
      double reservoir_width_;
      double flume_length_;
      double flume_width_;
    };
  } /* namespace Geometries */
} /* namespace ryujin */
