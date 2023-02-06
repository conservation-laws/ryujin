//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <hyperbolic_system.h>
#include <initial_state.h>

namespace ryujin
{
  namespace ShallowWater
  {
    /**
     * Circular dam break problem.
     *
     * @ingroup InitialValues
     */
    template <int dim, typename Number, typename state_type>
    class CircularDamBreak : public InitialState<dim, Number, state_type, 1>
    {
    public:
      CircularDamBreak(const HyperbolicSystem &hyperbolic_system,
                       const std::string sub)
          : InitialState<dim, Number, state_type, 1>("circular dam break", sub)
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

        return hyperbolic_system.template expand_state<dim>(
            HyperbolicSystem::state_type<1, Number>{{h, 0.}});
      }

      /* Default bathymetry of 0 */

    private:
      const HyperbolicSystem &hyperbolic_system;

      Number still_water_depth_;
      Number radius_;
      Number dam_amplitude_;
    };

  } // namespace ShallowWater
} // namespace ryujin
