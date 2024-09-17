//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// [LANL Copyright Statement]
// Copyright (C) 2024 by the ryujin authors
// Copyright (C) 2024 by Triad National Security, LLC
//

#pragma once

#include <initial_state_library.h>

namespace ryujin
{
  namespace EulerInitialStates
  {
    /**
     * An initial state that simulates an "intertial confinement fusion" like
     * problem. The set up consists of three regions: (i) a low density state
     * inside a perturbed interface; (ii) a high density state outside the
     * interface; (iii) an incoming shock wave characterized by its Mach number
     * and the state outside the interface as well as starting location (given
     * by a radius). The perturbed interface is characterized by the number of
     * modes and an amplitude.
     *
     * @ingroup EulerEquations
     */
    template <typename Description, int dim, typename Number>
    class ICFLike : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using View =
          typename Description::template HyperbolicSystemView<dim, Number>;
      using state_type = typename View::state_type;

      ICFLike(const HyperbolicSystem &hyperbolic_system,
              const std::string subsection)
          : InitialState<Description, dim, Number>("icf like", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        gamma_ = 1.4;
        if constexpr (!View::have_gamma) {
          this->add_parameter("gamma", gamma_, "The ratio of specific heats");
        }

        primitive_inside_[0] = 0.1;
        primitive_inside_[1] = 0.0;
        primitive_inside_[2] = 1.0;
        this->add_parameter(
            "primitive state inside",
            primitive_inside_,
            "Initial primitive state (rho, u, p) inside perturbed interface");

        primitive_outside_[0] = 1.0;
        primitive_outside_[1] = 0.0;
        primitive_outside_[2] = 1.0;
        this->add_parameter(
            "primitive state outside",
            primitive_outside_,
            "Initial primitive state (rho, u, p) outside perturbed interface");

        interface_radius_ = 1.0;
        this->add_parameter(
            "interface radius", interface_radius_, "Radius of interface");

        num_modes_ = 8.0;
        this->add_parameter("number of modes",
                            num_modes_,
                            "Number of modes for pertburation of interface");

        amplitude_ = 0.02;
        this->add_parameter(
            "amplitude", amplitude_, "Amplitude for interface pertburation");

        mach_number_ = 3.0;
        this->add_parameter(
            "mach number", mach_number_, "Mach number of incoming shock front");

        shock_radius_ = 1.2;
        this->add_parameter("shock radius",
                            shock_radius_,
                            "Radial location of incoming shock front");

        const auto convert_states = [&]() {
          const auto view = hyperbolic_system_.template view<dim, Number>();
          state_inside_ = view.from_initial_state(primitive_inside_);
          state_outside_ = view.from_initial_state(primitive_outside_);
        };
        this->parse_parameters_call_back.connect(convert_states);
        convert_states();
      };

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        const auto view = hyperbolic_system_.template view<dim, Number>();

        /* Compute incoming shock state */
        state_type conserved_shock_state;
        const auto r_hat = point / point.norm();
        {
          auto b = Number(0.);
          if constexpr (View::have_eos_interpolation_b)
            b = view.eos_interpolation_b();

          const auto &rho_R = primitive_outside_[0];
          const auto &u_R = primitive_outside_[1];
          const auto &p_R = primitive_outside_[2];
          /* a_R^2 = gamma * p / rho / (1 - b * rho) */
          const Number a_R = std::sqrt(gamma_ * p_R / rho_R / (1 - b * rho_R));
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

          state_type primitive_shock_state;
          primitive_shock_state[0] = rho_L;

          for (unsigned int i = 0; i < dim; ++i) {
            primitive_shock_state[i + 1] = 0.;
          }

          if (point.norm() > 0.) {
            for (unsigned int i = 0; i < dim; ++i) {
              primitive_shock_state[i + 1] = -u_L * r_hat[i];
            }
          }
          primitive_shock_state[1 + dim] = p_L;

          conserved_shock_state =
              view.from_initial_state(primitive_shock_state);
        }

        /* Define perturbation */
        const double angle = std::acos(std::abs(point[dim - 1]) / point.norm());

        const auto omega = num_modes_;
        const double perturbation = amplitude_ * std::cos(omega * angle);

        auto full_state =
            (point.norm() > interface_radius_ + perturbation ? state_outside_
                                                             : state_inside_);

        if (point.norm() > shock_radius_) {
          full_state = conserved_shock_state;
        }

        /* Set final state */
        return full_state;
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;

      Number gamma_;

      dealii::Tensor<1, 3, Number> primitive_inside_;
      dealii::Tensor<1, 3, Number> primitive_outside_;
      state_type state_inside_;
      state_type state_outside_;

      double interface_radius_;
      double num_modes_;
      double amplitude_;
      double shock_radius_;
      double mach_number_;
    };


  } // namespace EulerInitialStates
} // namespace ryujin
