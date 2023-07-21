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
     * An initial state formed by a contrast of a given "left" and "right"
     * primitive state.
     *
     * @note This class does not evolve a possible shock front in time. If
     * you need correct time-dependent Dirichlet data use @ref ShockFront
     * instead.
     *
     * @ingroup EulerEquations
     */
    template <typename Description, int dim, typename Number>
    class Contrast : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using HyperbolicSystemView =
          typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      Contrast(const HyperbolicSystemView &hyperbolic_system,
               const std::string subsection)
          : InitialState<Description, dim, Number>("contrast", subsection)
          , hyperbolic_system(hyperbolic_system)
      {
        primitive_left_[0] = 1.4;
        primitive_left_[1] = 0.;
        primitive_left_[2] = 1.;
        this->add_parameter(
            "primitive state left",
            primitive_left_,
            "Initial 1d primitive state (rho, u, p) on the left");

        primitive_right_[0] = 1.4;
        primitive_right_[1] = 0.;
        primitive_right_[2] = 1.;
        this->add_parameter(
            "primitive state right",
            primitive_right_,
            "Initial 1d primitive state (rho, u, p) on the right");

        // FIXME: update primitive
      }

      state_type compute(const dealii::Point<dim> &point, Number /*t*/) final
      {
        const auto &result = point[0] > 0. ? primitive_right_ : primitive_left_;
        return hyperbolic_system.from_primitive_state(
            hyperbolic_system.expand_state(result));
      }

    private:
      const HyperbolicSystemView hyperbolic_system;

      dealii::Tensor<1, 3, Number> primitive_left_;
      dealii::Tensor<1, 3, Number> primitive_right_;
    };
  } // namespace Euler
} // namespace ryujin
