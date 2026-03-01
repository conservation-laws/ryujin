//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// [LANL Copyright Statement]
// Copyright (C) 2024 by the ryujin authors
// Copyright (C) 2025 by Triad National Security, LLC
//

#pragma once

#include "hyperbolic_system.h"

#include <initial_state_library.h>
#include <iostream>

namespace ryujin
{
  namespace MultiSpeciesEulerInitialStates
  {
    using namespace MultiSpeciesEuler;

    /**
     * An initial state that simulates an "inertial confinement fusion" like
     * problem. The set up consists of three regions: (i) a low density state
     * inside a perturbed interface; (ii) a high density state outside the
     * interface; (iii) an incoming shock wave characterized by its Mach number
     * and the state outside the interface as well as starting location (given
     * by a radius). The perturbed interface is characterized by the number of
     * modes and an amplitude.
     *
     * @ingroup MultiSpeciesEulerEquations
     */
    template <typename Description, int dim, typename Number>
    class ICFLike : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using View =
          typename Description::template HyperbolicSystemView<dim, Number>;
      using state_type = typename View::state_type;

      static constexpr unsigned int primitive_dim = n_species + 2;
      using primitive_state_type = dealii::Tensor<1, primitive_dim, Number>;

      ICFLike(const HyperbolicSystem &hyperbolic_system,
              const std::string subsection)
          : InitialState<Description, dim, Number>("icf like", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        for (unsigned int k = 0; k < n_species - 1; ++k)
          temp_inside_[k] = Number(1.) / Number(n_species);
        temp_inside_[n_species - 1] = 1.0;
        temp_inside_[n_species] = 0.0;
        temp_inside_[n_species + 1] = 1.0;
        this->add_parameter("primitive state inside",
                            temp_inside_,
                            "Initial primitive state (Y_0, ..., Y_{n-2}, rho, "
                            "u, p) inside perturbed interface");

        for (unsigned int k = 0; k < n_species - 1; ++k)
          temp_outside_[k] = Number(1.) / Number(n_species);
        temp_outside_[n_species - 1] = 1.0;
        temp_outside_[n_species] = 0.0;
        temp_outside_[n_species + 1] = 1.0;
        this->add_parameter("primitive state outside",
                            temp_outside_,
                            "Initial primitive state (Y_0, ..., Y_{n-2}, rho, "
                            "u, p) outside perturbed interface");

        interface_radius_ = 1.0;
        this->add_parameter(
            "interface radius", interface_radius_, "Radius of interface");

        num_modes_ = 8.0;
        this->add_parameter("number of modes",
                            num_modes_,
                            "Number of modes for perturbation of interface");

        amplitude_ = 0.02;
        this->add_parameter(
            "amplitude", amplitude_, "Amplitude for interface perturbation");

        mach_number_ = 3.0;
        this->add_parameter(
            "mach number", mach_number_, "Mach number of incoming shock front");

        shock_radius_ = 1.2;
        this->add_parameter("shock radius",
                            shock_radius_,
                            "Radial location of incoming shock front");

        const auto convert_states = [&]() {
          const auto prim_inside = extend_primitive(temp_inside_);
          const auto prim_outside = extend_primitive(temp_outside_);

          const auto view = hyperbolic_system_.template view<dim, Number>();
          state_inside_ = view.from_initial_state(prim_inside);
          state_outside_ = view.from_initial_state(prim_outside);
        };
        this->parse_parameters_call_back.connect(convert_states);
        convert_states();
      };

      state_type compute(const dealii::Point<dim> &point, Number) final
      {
        const auto view = hyperbolic_system_.template view<dim, Number>();

        /* Compute incoming shock state */
        state_type conserved_shock_state;
        const auto r_hat = point / point.norm();
        {
          const auto gamma_ = view.gamma_mixture(state_outside_);

          const auto &rho_R = temp_outside_[n_species - 1];
          const auto &u_R = temp_outside_[n_species];
          const auto &p_R = temp_outside_[n_species + 1];

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

          state_type primitive_shock_state;

          /* Compute partial densities from mass fractions */
          Number Y_sum = Number(0.);
          for (unsigned int k = 0; k < n_species - 1; ++k) {
            primitive_shock_state[k] = temp_outside_[k] * rho_L;
            Y_sum += temp_outside_[k];
          }
          primitive_shock_state[n_species - 1] = (Number(1.) - Y_sum) * rho_L;

          /* Set radial velocity */
          for (unsigned int i = 0; i < dim; ++i) {
            primitive_shock_state[n_species + i] = Number(0.);
          }

          if (point.norm() > 0.) {
            for (unsigned int i = 0; i < dim; ++i) {
              primitive_shock_state[n_species + i] = -u_L * r_hat[i];
            }
          }
          primitive_shock_state[n_species + dim] = p_L;

          conserved_shock_state =
              view.from_initial_state(primitive_shock_state);
        }

        /* Compute polar (and potentially azimuthal) angle */
        const auto x = point[0];
        const auto y = dim > 1 ? point[1] : 0.;

        const double theta = std::atan2(y, x);

        double phi = 0.;
        if constexpr (dim == 3)
          phi = std::atan2(point[2], std::sqrt(x * x + y * y));

        /* Compute perturbation for interface */
        const auto omega = num_modes_;
        const double perturbation =
            amplitude_ * std::cos(omega * theta) * std::cos(omega * phi);

        /* Compute state depending on location */
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

      primitive_state_type temp_inside_;
      primitive_state_type temp_outside_;

      state_type state_inside_;
      state_type state_outside_;

      double interface_radius_;
      double num_modes_;
      double amplitude_;
      double shock_radius_;
      double mach_number_;

    };


  } // namespace MultiSpeciesEulerInitialStates
} // namespace ryujin
