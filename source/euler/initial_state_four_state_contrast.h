//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <initial_state_library.h>

namespace ryujin
{
  namespace EulerInitialStates
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
     * @ingroup EulerEquations
     */
    template <typename Description, int dim, typename Number>
    class FourStateContrast : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using HyperbolicSystemView =
          typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      FourStateContrast(const HyperbolicSystemView &hyperbolic_system,
                        const std::string &subsection)
          : InitialState<Description, dim, Number>("four state contrast",
                                                   subsection)
          , hyperbolic_system(hyperbolic_system)
      {

        /* Set default values and get primitive states from user */
        primitive_top_left_[0] = 1.4;
        primitive_top_left_[1] = 0.;
        primitive_top_left_[2] = 0.;
        primitive_top_left_[3] = 1.;

        this->add_parameter(
            "primitive state top left",
            primitive_top_left_,
            "Initial primitive state (rho, u, v, p) on top left");

        primitive_top_right_[0] = 1.4;
        primitive_top_right_[1] = 0.;
        primitive_top_right_[2] = 0.;
        primitive_top_right_[3] = 1.;
        this->add_parameter(
            "primitive state top right",
            primitive_top_right_,
            "Initial primitive state (rho, u, v, p) on top right");

        primitive_bottom_right_[0] = 1.4;
        primitive_bottom_right_[1] = 0.;
        primitive_bottom_right_[2] = 0.;
        primitive_bottom_right_[3] = 1.;
        this->add_parameter(
            "primitive state bottom right",
            primitive_bottom_right_,
            "Initial primitive state (rho, u, v, p) on bottom right");

        primitive_bottom_left_[0] = 1.4;
        primitive_bottom_left_[1] = 0.;
        primitive_bottom_left_[2] = 0.;
        primitive_bottom_left_[3] = 1.;
        this->add_parameter(
            "primitive state bottom left",
            primitive_bottom_left_,
            "Initial primitive state (rho, u, v, p) on bottom left");

        // FIXME: update primitive
      }

      state_type compute(const dealii::Point<dim> &point, Number /*t*/) final
      {
        if constexpr (dim == 1) {
          AssertThrow(false, dealii::ExcNotImplemented());
          __builtin_trap();

        } else {

          const auto top =
              point[0] >= 0. ? primitive_top_right_ : primitive_top_left_;
          const auto bottom =
              point[0] >= 0. ? primitive_bottom_right_ : primitive_bottom_left_;
          const auto result = point[1] >= 0. ? top : bottom;

          return hyperbolic_system.from_primitive_state(
              hyperbolic_system.expand_state(result));
        }
      }

    private:
      const HyperbolicSystemView hyperbolic_system;

      dealii::Tensor<1, 4, Number> primitive_top_left_;
      dealii::Tensor<1, 4, Number> primitive_bottom_left_;
      dealii::Tensor<1, 4, Number> primitive_top_right_;
      dealii::Tensor<1, 4, Number> primitive_bottom_right_;
    };
  } // namespace Euler
} // namespace ryujin
