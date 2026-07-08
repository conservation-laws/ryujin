//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// [LANL Copyright Statement]
// Copyright (C) 2024 - 2025 by the ryujin authors
// Copyright (C) 2024 by Triad National Security, LLC
//

#pragma once

#include <compile_time_options.h>

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
      using state_type_1d = typename Description::
          template HyperbolicSystemView<1, Number>::state_type;

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
        this->add_parameter("primitive state inside",
                            primitive_inside_,
                            "1d primitive state [rho, u, p] (for the "
                            "Noble-Abel gas EOS) inside perturbed interface");

        primitive_outside_[0] = 1.0;
        primitive_outside_[1] = 0.0;
        primitive_outside_[2] = 1.0;
        this->add_parameter("primitive state outside",
                            primitive_outside_,
                            "1d primitive state [rho, u, p] (for the "
                            "Noble-Abel gas EOS) outside perturbed interface");

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

          using state_type_1d = typename Description::
              template HyperbolicSystemView<1, Number>::state_type;
          static_assert(state_type_1d::dimension <=
                        dealii::Tensor<1, 3, Number>::dimension);

          state_type_1d result_inside;
          state_type_1d result_outside;
          for (unsigned int i = 0; i < state_type_1d::dimension; ++i) {
            result_inside[i] = primitive_inside_[i];
            result_outside[i] = primitive_outside_[i];
          }
          state_inside_ = view.from_initial_state(result_inside);
          state_outside_ = view.from_initial_state(result_outside);
        };
        this->parse_parameters_call_back.connect(convert_states);
        convert_states();
      };

      state_type compute(const dealii::Point<dim> &point, Number) final
      {
        const auto view = hyperbolic_system_.template view<dim, Number>();

        /* Compute polar (and potentially azimuthal) angle: */
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

        if (point.norm() > shock_radius_) {
          /*
           * Inside the incoming shock front:
           */

          const auto r_hat = point / point.norm();

          auto b = Number(0.);
          if constexpr (View::have_covolume_constant)
            b = view.eos_covolume_constant();

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
          if constexpr (View::have_energy_equation)
            primitive_shock_state[1 + dim] = p_L;

          return view.from_initial_state(primitive_shock_state);

        } else if (point.norm() > interface_radius_ + perturbation) {
          /*
           * Outside annulus between inner disc and outer shock annulus:
           */

          return state_outside_;

        } else {
          /*
           * Inner disc:
           */

          return state_inside_;
        }
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
