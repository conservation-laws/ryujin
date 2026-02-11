//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2024 by the ryujin authors
// // Copyright (C) 2025 by Triad National Security, LLC
//

#pragma once

#include <initial_state_library.h>

namespace ryujin
{
  namespace MultiSpeciesEulerInitialStates
  {
    /**
     * A shock-bubble initial state. The shock HAS to be initiated for when
     * either the mass fraction is 0 or 1 ONLY. The implementation assumes we
     * are working in the direction (1, ..., 0).
     *
     * @note The @p t argument is ignored. This class always returns the
     * initial configuration.
     *
     * @ingroup MultiSpeciesEulerEquations
     */
    template <typename Description, int dim, typename Number>
    class ShockBubble : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using View =
          typename Description::template HyperbolicSystemView<dim, Number>;
      using state_type = typename View::state_type;

      ShockBubble(const HyperbolicSystem &hyperbolic_system,
                  const std::string subsection)
          : InitialState<Description, dim, Number>("shock bubble", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        temp_inside_[0] = 0.5;
        temp_inside_[1] = 1.0;
        temp_inside_[2] = 0.0;
        temp_inside_[3] = 1.0;
        this->add_parameter(
            "primitive state bubble",
            temp_inside_,
            "Initial 1d primitive state (Y_0, rho, u, p) inside bubble");

        temp_ambient_[0] = 0.5;
        temp_ambient_[1] = 1.0;
        temp_ambient_[2] = 0.0;
        temp_ambient_[3] = 1.0;
        this->add_parameter(
            "primitive state pre-shock",
            temp_ambient_,
            "Initial 1d primitive state (Y_0, rho, u, p) ambient pre-shock");

        radius_ = 1.0;
        this->add_parameter("radius", radius_, "Bubble radius");

        bubble_center_[0] = 0.;
        if constexpr (dim > 1)
          bubble_center_[1] = 0.;

        this->add_parameter(
            "bubble center", bubble_center_, "The dim sized bubble center");

        shock_location_ = 0.0;
        this->add_parameter("shock distance from bubble",
                            shock_location_,
                            "The distance from the bubble center. Negative "
                            "value to the left, positive value to the right. ");

        mach_number_ = 2.0;
        this->add_parameter(
            "mach number", mach_number_, "Mach number of shock front ");

        const auto convert_states = [&]() {
          const auto view = hyperbolic_system_.template view<dim, Number>();

          primitive_inside_ = extend(temp_inside_);
          primitive_ambient_ = extend(temp_ambient_);

          state_bubble_ = view.from_initial_state(primitive_inside_);
          state_ambient_ = view.from_initial_state(primitive_ambient_);
        };
        this->parse_parameters_call_back.connect(convert_states);
        convert_states();
      }


      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        const auto view = hyperbolic_system_.template view<dim, Number>();

        const auto &x = point[0];
        const auto xbar = point - bubble_center_;
        const auto &xc = bubble_center_[0];
        auto sign = 1.;
        if (xc < shock_location_)
          sign = -1.;

        /* Compute shocked state outside */
        {
          const auto gamma_ = view.gamma_mixture(state_ambient_);

          const auto &rho_R = temp_ambient_[1];
          const auto &u_R = temp_ambient_[2];
          const auto &p_R = temp_ambient_[3];
          /* a_R^2 = gamma * p / rho / (1 - b * rho) */
          const Number a_R = std::sqrt(gamma_ * p_R / rho_R);
          const Number mach_R = u_R / a_R;

          auto S3_ = mach_number_ * a_R;
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

          const auto Y0_outside = temp_ambient_[0];

          primitive_shock_[0] = Y0_outside;
          primitive_shock_[1] = rho_L;
          primitive_shock_[2] = sign * u_L;
          primitive_shock_[3] = p_L;

          primitive_shock_ = extend(primitive_shock_);
          state_shock_ = view.from_initial_state(primitive_shock_);
        }

        auto final_state =
            (xbar.norm() <= radius_ ? state_bubble_ : state_ambient_);

        if (x <= xc + shock_location_)
          final_state = state_shock_;

        /* Set final state */
        return final_state;
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;

      dealii::Tensor<1, 4, Number> temp_inside_;
      dealii::Tensor<1, 4, Number> temp_ambient_;

      dealii::Tensor<1, 4, Number> primitive_inside_;
      dealii::Tensor<1, 4, Number> primitive_ambient_;
      dealii::Tensor<1, 4, Number> primitive_shock_;

      state_type state_bubble_;
      state_type state_ambient_;
      state_type state_shock_;

      dealii::Point<dim> bubble_center_;

      double radius_;
      double mach_number_;
      double shock_location_;

      DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, 4, Number>
      extend(dealii::Tensor<1, 4, Number> &temp_in) const
      {
        dealii::Tensor<1, 4, Number> result;
        result[0] = temp_in[0] * temp_in[1];        // = alpha_0 rho_0;
        result[1] = (1. - temp_in[0]) * temp_in[1]; // = alpha_1 rho_1;

        for (unsigned int i = 2; i < 4; ++i)
          result[i] = temp_in[i];

        return result;
      }
    };


  } // namespace MultiSpeciesEulerInitialStates
} // namespace ryujin
