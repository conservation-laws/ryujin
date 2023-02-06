//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <hyperbolic_system.h>
#include <initial_state.h>

namespace ryujin
{
  namespace EulerAEOS
  {
    /**
     * An initial state formed by a contrast of a given "left" and "right"
     * primitive state.
     *
     * @note This class does not evolve a possible shock front in time. If
     * you need correct time-dependent Dirichlet data use @ref ShockFront
     * instead.
     *
     * @ingroup InitialValues
     */
    template <int dim, typename Number, typename state_type>
    class Contrast : public InitialState<dim, Number, state_type>
    {
    public:
      Contrast(const HyperbolicSystem &hyperbolic_system,
               const std::string subsection)
          : InitialState<dim, Number, state_type>("contrast", subsection)
          , hyperbolic_system(hyperbolic_system)
      {
        primitive_left_[0] = 7. / 5.;
        primitive_left_[1] = 0.;
        primitive_left_[2] = 1. / .4;
        this->add_parameter(
            "primitive state left",
            primitive_left_,
            "Initial 1d primitive state (rho, u, e) on the left");

        primitive_right_[0] = 7. / 5.;
        primitive_right_[1] = 0.;
        primitive_right_[2] = 1. / .4;
        this->add_parameter(
            "primitive state right",
            primitive_right_,
            "Initial 1d primitive state (rho, u, e) on the right");
      }

      state_type compute(const dealii::Point<dim> &point, Number /*t*/) final
      {
        const auto temp = hyperbolic_system.from_primitive_state(
            point[0] > 0. ? primitive_right_ : primitive_left_);
        return hyperbolic_system.template expand_state<dim>(temp);
      }

    private:
      const HyperbolicSystem &hyperbolic_system;

      dealii::Tensor<1, 3, Number> primitive_left_;
      dealii::Tensor<1, 3, Number> primitive_right_;
    };
  } // namespace EulerAEOS
} // namespace ryujin
