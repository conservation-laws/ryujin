//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <initial_state.h>

#include <deal.II/base/tensor.h>

namespace ryujin
{
  namespace InitialStateLibrary
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
      CircularDamBreak(const HyperbolicSystem & /*hyperbolic_system*/,
                       const std::string sub)
          : InitialState<dim, Number, state_type, 1>("circular dam break", sub)
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

      virtual state_type compute(const dealii::Point<dim> &point,
                                 Number /*t*/) final override
      {
        if constexpr (dim == 1) {
          AssertThrow(false, dealii::ExcNotImplemented());
          return state_type();
        }

        const Number r = point.norm_square();
        const Number h = (r <= radius_ ? dam_amplitude_ : still_water_depth_);

        return state_type({h, 0., 0.});
      }

      /* Default bathymetry of 0 */

    private:
      Number still_water_depth_;
      Number radius_;
      Number dam_amplitude_;
    };

  } // namespace InitialStates
} // namespace ryujin
