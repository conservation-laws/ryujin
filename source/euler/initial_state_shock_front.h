//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include <initial_state_library.h>

namespace ryujin
{
  namespace EulerInitialStates
  {
    /**
     * The S1/S3 shock front.
     *
     * An Analytic solution for the compressible Euler equations with
     * polytropic gas equation of state consisting of a shock front
     * evolving in time.
     *
     * @note This class returns the analytic solution as a function of time
     * @p t and position @p x.
     *
     * @ingroup EulerEquations
     */
    template <typename Description, int dim, typename Number>
    class ShockFront : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using HyperbolicSystemView =
          typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      ShockFront(const HyperbolicSystem &hyperbolic_system,
                 const std::string subsection)
          : InitialState<Description, dim, Number>("shock front", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        gamma_ = 1.4;
        if constexpr (!HyperbolicSystemView::have_gamma) {
          this->add_parameter("gamma", gamma_, "The ratio of specific heats");
        }

        primitive_right_[0] = 1.4;
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

        const auto compute_and_convert_states = [&]() {
          if constexpr (HyperbolicSystemView::have_gamma) {
            gamma_ = hyperbolic_system_.gamma();
          }

          /* Compute post-shock state and S3: */

          auto b = Number(0.);
          if constexpr (HyperbolicSystemView::have_eos_interpolation_b)
            b = hyperbolic_system_.eos_interpolation_b();

          const auto &rho_R = primitive_right_[0];
          const auto &u_R = primitive_right_[1];
          const auto &p_R = primitive_right_[2];
          /* a_R^2 = gamma * p / rho / (1 - b * rho) */
          const Number a_R = std::sqrt(gamma_ * p_R / rho_R / (1 - b * rho_R));
          const Number mach_R = u_R / a_R;

          S3_ = mach_number_ * a_R;
          const Number delta_mach = mach_R - mach_number_;

          const Number rho_L =
              rho_R * (gamma_ + Number(1.)) * delta_mach * delta_mach /
              ((gamma_ - Number(1.)) * delta_mach * delta_mach + Number(2.));
          const Number u_L =
              (Number(1.) - rho_R / rho_L) * S3_ + rho_R / rho_L * u_R;
          const Number p_L = p_R *
                             (Number(2.) * gamma_ * delta_mach * delta_mach -
                              (gamma_ - Number(1.))) /
                             (gamma_ + Number(1.));

          primitive_left_[0] = rho_L;
          primitive_left_[1] = u_L;
          primitive_left_[2] = p_L;

          state_left_ = hyperbolic_system_.from_initial_state(primitive_left_);
          state_right_ =
              hyperbolic_system_.from_initial_state(primitive_right_);
        };

        this->parse_parameters_call_back.connect(compute_and_convert_states);
        compute_and_convert_states();
      }


      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        const Number position_1d = Number(point[0] - S3_ * t);
        return (position_1d > 0. ? state_right_ : state_left_);
      }

    private:
      const HyperbolicSystemView hyperbolic_system_;

      Number gamma_;

      dealii::Tensor<1, 3, Number> primitive_left_;
      dealii::Tensor<1, 3, Number> primitive_right_;
      Number mach_number_;
      Number S3_;

      state_type state_left_;
      state_type state_right_;
    };
  } // namespace EulerInitialStates
} // namespace ryujin
