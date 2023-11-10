//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"
#include <initial_state_library.h>

namespace ryujin
{
  namespace ShallowWater
  {
    struct Description;

    /**
     * Dam break with flat portion followed by a sloping ramp.
     *
     * @ingroup ShallowWaterEquations
     */
    template <int dim, typename Number>
    class SlopingRampDamBreak : public InitialState<Description, dim, Number>
    {
      using HyperbolicSystemView = HyperbolicSystem::View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;
      using primitive_state_type =
          typename HyperbolicSystemView::primitive_state_type;

    public:
      SlopingRampDamBreak(const HyperbolicSystem &hyperbolic_system,
                          const std::string s)
          : InitialState<Description, dim, Number>("sloping ramp", s)
          , hyperbolic_system(hyperbolic_system)
      {
        inflow = false;
        this->add_parameter(
            "inflow",
            inflow,
            "If set to true then a constant inflow is computed for t>0 "
            "suitable for prescribing Dirichlet conditions at the inflow "
            "boundary.");


        left_depth = 1.875;
        this->add_parameter("left water depth",
                            left_depth,
                            "Depth of water to the left of pseudo-dam");
        right_depth = 0.;
        this->add_parameter("right water depth",
                            right_depth,
                            "Depth of water to the right of pseudo-dam");

        ramp_slope = 1.;
        this->add_parameter(
            "ramp slope", ramp_slope, "To modify slope of ramp");

        flat_length = 5.;
        this->add_parameter("length of flat part",
                            flat_length,
                            "To modify length of flat portion");
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        const Number x = point[0];

        /* Initial state: */

        if (t <= 1.e-10 || !inflow) {
          Number h = x < 0 ? left_depth : right_depth;
          h = std::max(h - compute_bathymetry(point), Number(0.));
          return state_type{{h, 0.}};
        }

        /* For t > 0 prescribe constant inflow Dirichlet data on the left: */

        const auto &h = left_depth;
        const auto a =
            hyperbolic_system.speed_of_sound(state_type{{h, Number(0.)}});

        return state_type{{h, h * a}};
      }

      auto initial_precomputations(const dealii::Point<dim> &point) ->
          typename InitialState<Description, dim, Number>::
              precomputed_state_type final
      {
        /* Compute bathymetry: */
        return {compute_bathymetry(point)};
      }

    private:
      const HyperbolicSystemView hyperbolic_system;

      DEAL_II_ALWAYS_INLINE inline Number
      compute_bathymetry(const dealii::Point<dim> &point) const
      {

        const Number &x = point[0];
        return std::max(ramp_slope * (x - flat_length), Number(0.));
      }

      bool inflow;
      Number left_depth;
      Number right_depth;
      Number flat_length;
      Number ramp_slope;
    };

  } // namespace ShallowWater
} // namespace ryujin
