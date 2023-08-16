//
// SPDX-License-Identifier: MIT or BSD-3-Clause
// [LANL Copyright Statement]
// Copyright (C) 2020 - 2023 by the ryujin authors
// Copyright (C) 2023 - 2023 by Triad National Security, LLC
//

#pragma once

#include <initial_state_library.h>

namespace ryujin
{
  namespace ShallowWaterInitialStates
  {
    /**
     * An initial state formed by a contrast of a given "left" and "right"
     * primitive state.
     *
     * @ingroup ShallowWaterEquations
     */
    template <typename Description, int dim, typename Number>
    class Contrast : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using HyperbolicSystemView =
          typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      Contrast(const HyperbolicSystem &hyperbolic_system,
               const std::string subsection)
          : InitialState<Description, dim, Number>("contrast", subsection)
          , hyperbolic_system_(hyperbolic_system)
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
        return hyperbolic_system_.from_initial_state(temp);
      }

    private:
      const HyperbolicSystemView hyperbolic_system_;

      dealii::Tensor<1, 2, Number> primitive_left_;
      dealii::Tensor<1, 2, Number> primitive_right_;
    };
  } // namespace ShallowWaterInitialStates
} // namespace ryujin
