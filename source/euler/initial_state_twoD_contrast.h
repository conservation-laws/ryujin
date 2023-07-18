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
    struct Description;

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
     * @ingroup EulerEquations
     */
    template <int dim, typename Number>
    class TwoDContrast : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystemView = HyperbolicSystem::View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      TwoDContrast(const HyperbolicSystemView &hyperbolic_system,
                   const std::string &subsection)
          : InitialState<Description, dim, Number>("2d contrast", subsection)
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
        if constexpr (dim != 2) {
          AssertThrow(false, dealii::ExcNotImplemented());
          __builtin_trap();
        }

        /* Set temporary states depending on location */
        const auto temp_top =
            point[0] >= 0. ? primitive_top_right_ : primitive_top_left_;
        const auto temp_bottom =
            point[0] >= 0. ? primitive_bottom_right_ : primitive_bottom_left_;

        return hyperbolic_system.from_primitive_state(
            point[1] >= 0. ? temp_top : temp_bottom);
      }

    private:
      const HyperbolicSystemView hyperbolic_system;

      state_type primitive_top_left_;
      state_type primitive_bottom_left_;
      state_type primitive_top_right_;
      state_type primitive_bottom_right_;
    };
  } // namespace Euler
} // namespace ryujin
