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
     * An S1/S3 shock front
     *
     * @todo Documentation
     *
     * @ingroup EulerEquations
     */
    template <int dim, typename Number, typename state_type>
    class ShockFront : public InitialState<dim, Number, state_type>
    {
    public:
      ShockFront(const HyperbolicSystem &hyperbolic_system,
                 const std::string subsection)
          : InitialState<dim, Number, state_type>("shockfront", subsection)
          , hyperbolic_system(hyperbolic_system)
      {
        dealii::ParameterAcceptor::parse_parameters_call_back.connect(std::bind(
            &ShockFront<dim, Number, state_type>::parse_parameters_callback,
            this));

        primitive_right_[0] = hyperbolic_system.gamma();
        primitive_right_[1] = 0.0;
        primitive_right_[2] = 1.;
        this->add_parameter("primitive state",
                            primitive_right_,
                            "Initial 1d primitive state (rho, u, p) before the "
                            "shock (to the right)");

        mach_number_ = 2.0;
        this->add_parameter(
            "mach number",
            mach_number_,
            "Mach number of shock front (S1, S3 = mach * a_L/R)");
      }

      void parse_parameters_callback()
      {
        /* Compute post-shock state and S3: */

        const auto gamma = hyperbolic_system.gamma();
        const Number b = Number(0.); // FIXME

        const auto &rho_R = primitive_right_[0];
        const auto &u_R = primitive_right_[1];
        const auto &p_R = primitive_right_[2];
        /* a_R^2 = gamma * p / rho / (1 - b * rho) */
        const Number a_R = std::sqrt(gamma * p_R / rho_R / (1 - b * rho_R));
        const Number mach_R = u_R / a_R;

        S3_ = mach_number_ * a_R;
        const Number delta_mach = mach_R - mach_number_;

        const Number rho_L =
            rho_R * (gamma + Number(1.)) * delta_mach * delta_mach /
            ((gamma - Number(1.)) * delta_mach * delta_mach + Number(2.));
        const Number u_L =
            (Number(1.) - rho_R / rho_L) * S3_ + rho_R / rho_L * u_R;
        const Number p_L = p_R *
                           (Number(2.) * gamma * delta_mach * delta_mach -
                            (gamma - Number(1.))) /
                           (gamma + Number(1.));

        primitive_left_[0] = rho_L;
        primitive_left_[1] = u_L;
        primitive_left_[2] = p_L;
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        const Number position_1d = Number(point[0] - S3_ * t);

        const auto temp = hyperbolic_system.from_primitive_state(
            position_1d > 0. ? primitive_right_ : primitive_left_);
        return hyperbolic_system.template expand_state<dim>(temp);
      }

    private:
      const HyperbolicSystem &hyperbolic_system;

      dealii::Tensor<1, 3, Number> primitive_left_;
      dealii::Tensor<1, 3, Number> primitive_right_;
      Number mach_number_;
      Number S3_;
    };
  } // namespace Euler
} // namespace ryujin
