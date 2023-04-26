//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"
#include <initial_state.h>

namespace ryujin
{
  namespace Euler
  {
    /**
     * A 2D extension of the "contrast" initial state consisting of 4 different
     * states separated at x = 0 and y = 0. Visually, this looks like:
     *
     *        state 1  | state 2
     *        ---------|-----------
     *        state 3  | state 4
     *
     * @note This class does not evolve a possible shock front in time. If
     * you need correct time-dependent Dirichlet data use @ref ShockFront
     * instead.
     *
     * @ingroup InitialValues
     */
    template <int dim, typename Number, typename state_type>
    class TwoDContrast : public InitialState<dim, Number, state_type>
    {
    public:
      TwoDContrast(const HyperbolicSystem &hyperbolic_system,
                   const std::string subsection)
          : InitialState<dim, Number, state_type>("2d contrast", subsection)
          , hyperbolic_system(hyperbolic_system)
      {
        /* Set default values and get primitive states from user */
        primitive_top_left_[0] = 1.0;
        primitive_top_left_[1] = 0.0;
        primitive_top_left_[2] = 0.0;
        primitive_top_left_[3] = 1.;

        this->add_parameter(
            "primitive state top left",
            primitive_top_left_,
            "Initial primitive state (rho, u, v, p) on top left");

        primitive_top_right_[0] = 0.125;
        primitive_top_right_[1] = 0.0;
        primitive_top_right_[2] = 0.0;
        primitive_top_right_[3] = 0.1;
        this->add_parameter(
            "primitive state top right",
            primitive_top_right_,
            "Initial primitive state (rho, u, v, p) on top right");

        primitive_bottom_right_[0] = 1.0;
        primitive_bottom_right_[1] = 0.0;
        primitive_bottom_right_[2] = 0.0;
        primitive_bottom_right_[3] = 0.1;
        this->add_parameter(
            "primitive state bottom right",
            primitive_bottom_right_,
            "Initial primitive state (rho, u, v, p) on bottom right");

        primitive_bottom_left_[0] = 1.0;
        primitive_bottom_left_[1] = 0.0;
        primitive_bottom_left_[2] = 0.0;
        primitive_bottom_left_[3] = 1.;
        this->add_parameter(
            "primitive state bottom left",
            primitive_bottom_left_,
            "Initial primitive state (rho, u, v, p) on bottom left");
      }

      state_type compute(const dealii::Point<dim> &point, Number /*t*/) final
      {
        /* Set temporary states depending on location */
        auto temp_top =
            point[0] >= 0. ? primitive_top_right_ : primitive_top_left_;

        auto temp_bottom =
            point[0] >= 0. ? primitive_bottom_right_ : primitive_bottom_left_;

        /* Convert to regular states */
        temp_top = hyperbolic_system.from_primitive_state(temp_top);
        temp_bottom = hyperbolic_system.from_primitive_state(temp_bottom);

        /* Return final state if dim = 2 */
        if constexpr (dim != 2) {
          AssertThrow(false, dealii::ExcNotImplemented());
          __builtin_trap();
        } else
          return hyperbolic_system.template expand_state<dim>(
              point[1] >= 0. ? temp_top : temp_bottom);
      }

    private:
      const HyperbolicSystem &hyperbolic_system;

      dealii::Tensor<1, 4, Number> primitive_top_left_;
      dealii::Tensor<1, 4, Number> primitive_bottom_left_;
      dealii::Tensor<1, 4, Number> primitive_top_right_;
      dealii::Tensor<1, 4, Number> primitive_bottom_right_;
    };
  } // namespace Euler
} // namespace ryujin
