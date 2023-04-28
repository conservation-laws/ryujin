//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"
#include <initial_state.h>

namespace ryujin
{
  namespace ShallowWater
  {
    /**
     * An initial state formed by a contrast of a given "left" and "right"
     * primitive state.
     *
     * @ingroup ShallowWaterEquations
     */
    template <int dim, typename Number, typename state_type>
    class Contrast : public InitialState<dim, Number, state_type, 1>
    {
    public:
      Contrast(const HyperbolicSystem &hyperbolic_system,
               const std::string subsection)
          : InitialState<dim, Number, state_type, 1>("contrast", subsection)
          , hyperbolic_system(hyperbolic_system)
      {
        primitive_left_[0] = 1.;
        primitive_left_[1] = 0.0;
        this->add_parameter("primitive state left",
                            primitive_left_,
                            "Initial 1d primitive state (h, u) on the left");

        primitive_right_[0] = 1.;
        primitive_right_[1] = 0.0;
        this->add_parameter("primitive state right",
                            primitive_right_,
                            "Initial 1d primitive state (h, u) on the right");
      }

      state_type compute(const dealii::Point<dim> &point, Number /*t*/) final
      {
        const auto temp = hyperbolic_system.from_primitive_state(
            point[0] > 0. ? primitive_right_ : primitive_left_);
        return hyperbolic_system.template expand_state<dim>(temp);
      }

    private:
      const HyperbolicSystem &hyperbolic_system;

      dealii::Tensor<1, 2, Number> primitive_left_;
      dealii::Tensor<1, 2, Number> primitive_right_;
    };
  } // namespace ShallowWater
} // namespace ryujin
