//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2024 by the ryujin authors
// // Copyright (C) 2025 by Triad National Security, LLC
//

#pragma once

#include "hyperbolic_system.h"

#include <initial_state_library.h>

namespace ryujin
{
  namespace MultiSpeciesEulerInitialStates
  {
    using namespace MultiSpeciesEuler;

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

      static constexpr unsigned int primitive_dim = n_species + 2;
      using primitive_state_type = dealii::Tensor<1, primitive_dim, Number>;

      ShockBubble(const HyperbolicSystem &hyperbolic_system,
                  const std::string subsection)
          : InitialState<Description, dim, Number>("shock bubble", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        for (unsigned int k = 0; k < n_species - 1; ++k)
          temp_inside_[k] = Number(1.) / Number(n_species);
        temp_inside_[n_species - 1] = 1.0;
        temp_inside_[n_species] = 0.0;
        temp_inside_[n_species + 1] = 1.0;
        this->add_parameter(
            "primitive state bubble",
            temp_inside_,
            "Initial 1d primitive state (Y_0, ..., Y_{n-2}, rho, u, p) inside "
            "bubble");

        for (unsigned int k = 0; k < n_species - 1; ++k)
          temp_ambient_[k] = Number(1.) / Number(n_species);
        temp_ambient_[n_species - 1] = 1.0;
        temp_ambient_[n_species] = 0.0;
        temp_ambient_[n_species + 1] = 1.0;
        this->add_parameter(
            "primitive state pre-shock",
            temp_ambient_,
            "Initial 1d primitive state (Y_0, ..., Y_{n-2}, rho, u, p) ambient "
            "pre-shock");

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

          const auto primitive_inside = extend_primitive(temp_inside_);
          const auto primitive_ambient = extend_primitive(temp_ambient_);

          state_bubble_ = view.from_initial_state(primitive_inside);
          state_ambient_ = view.from_initial_state(primitive_ambient);

          /* Precompute shocked state via Rankine-Hugoniot: */

          const auto gamma_ = view.gamma_mixture(state_ambient_);

          const auto &rho_R = temp_ambient_[n_species - 1];
          const auto &u_R = temp_ambient_[n_species];
          const auto &p_R = temp_ambient_[n_species + 1];

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

          auto sign = 1.;
          if (bubble_center_[0] < shock_location_)
            sign = -1.;

          primitive_state_type prim_shock;
          for (unsigned int k = 0; k < n_species - 1; ++k)
            prim_shock[k] = temp_ambient_[k];
          prim_shock[n_species - 1] = rho_L;
          prim_shock[n_species] = sign * u_L;
          prim_shock[n_species + 1] = p_L;

          state_shock_ = view.from_initial_state(extend_primitive(prim_shock));
        };
        this->parse_parameters_call_back.connect(convert_states);
        convert_states();
      }


      state_type compute(const dealii::Point<dim> &point, Number /*t*/) final
      {
        const auto xbar = point - bubble_center_;

        auto final_state =
            (xbar.norm() <= radius_ ? state_bubble_ : state_ambient_);

        if (point[0] <= bubble_center_[0] + shock_location_)
          final_state = state_shock_;

        return final_state;
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;

      primitive_state_type temp_inside_;
      primitive_state_type temp_ambient_;

      state_type state_bubble_;
      state_type state_ambient_;
      state_type state_shock_;

      dealii::Point<dim> bubble_center_;

      double radius_;
      double mach_number_;
      double shock_location_;

    };


  } // namespace MultiSpeciesEulerInitialStates
} // namespace ryujin
