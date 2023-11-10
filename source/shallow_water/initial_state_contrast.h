//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
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
     * An initial state formed by a contrast of a given "left" and "right"
     * primitive state.
     *
     * @ingroup ShallowWaterEquations
     */
    template <int dim, typename Number>
    class Contrast : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystemView = HyperbolicSystem::View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;
      using primitive_state_type =
          typename HyperbolicSystemView::primitive_state_type;

      Contrast(const HyperbolicSystem &hyperbolic_system,
               const std::string subsection)
          : InitialState<Description, dim, Number>("contrast", subsection)
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
        const auto temp = point[0] > 0. ? primitive_right_ : primitive_left_;
        return hyperbolic_system.from_primitive_state(
            hyperbolic_system.expand_state(temp));
      }

    private:
      const HyperbolicSystemView hyperbolic_system;

      dealii::Tensor<1, 2, Number> primitive_left_;
      dealii::Tensor<1, 2, Number> primitive_right_;
    };
  } // namespace ShallowWater
} // namespace ryujin
