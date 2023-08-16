//
// SPDX-License-Identifier: MIT or BSD-3-Clause
// [LANL Copyright Statement]
// Copyright (C) 2020 - 2023 by the ryujin authors
// Copyright (C) 2023 - 2023 by Triad National Security, LLC
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
     * Circular dam break problem.
     *
     * @ingroup ShallowWaterEquations
     */
    template <int dim, typename Number>
    class CircularDamBreak : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystemView = HyperbolicSystem::View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;
      using primitive_state_type =
          typename HyperbolicSystemView::primitive_state_type;

      CircularDamBreak(const HyperbolicSystem &hyperbolic_system,
                       const std::string sub)
          : InitialState<Description, dim, Number>("circular dam break", sub)
          , hyperbolic_system(hyperbolic_system)
      {
        still_water_depth_ = 0.5;
        this->add_parameter("still water depth",
                            still_water_depth_,
                            "Depth of still water outside circular dam");
        radius_ = 2.5;
        this->add_parameter("radius", radius_, "Radius of circular dam ");

        dam_amplitude_ = 2.5;
        this->add_parameter(
            "dam amplitude", dam_amplitude_, "Amplitude of circular dam");
      }

      state_type compute(const dealii::Point<dim> &point, Number /*t*/) final
      {
        const Number r = point.norm_square();
        const Number h = (r <= radius_ ? dam_amplitude_ : still_water_depth_);

        return state_type{{h, 0.}};
      }

      /* Default bathymetry of 0 */

    private:
      const HyperbolicSystemView hyperbolic_system;

      Number still_water_depth_;
      Number radius_;
      Number dam_amplitude_;
    };

  } // namespace ShallowWater
} // namespace ryujin
