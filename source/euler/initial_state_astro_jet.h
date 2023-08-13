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
     * The astrophysical jet problem introduced in @cite Zhang-Shu-2010 without
     * radiative cooling. General set up in standard SI units is:
     *
     * Ambient state: (rho, u, p) = (5,  0,  0.4127);
     * Mach 80 jet:   (rho, u, p) = (5,  30, 0.4127), T = 0.07;
     * Mach 200 jet:  (rho, u, p) = (5, 800, 0.4127), T = 0.001;
     *
     * See section 4.4 of reference for more details.
     *
     * @ingroup EulerEquations
     */
    template <typename Description, int dim, typename Number>
    class AstroJet : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using HyperbolicSystemView =
          typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      AstroJet(const HyperbolicSystem &hyperbolic_system,
               const std::string subsection)
          : InitialState<Description, dim, Number>("astro jet", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        gamma_ = 5. / 3.;
        if constexpr (!HyperbolicSystemView::have_gamma) {
          this->add_parameter("gamma", gamma_, "The ratio of specific heats");
        }

        jet_width_ = 0.05;
        this->add_parameter("jet width",
                            jet_width_,
                            "The width of the jet coming out of boundary");

        jet_state_[0] = 5.0;
        jet_state_[1] = 30.0;
        jet_state_[2] = 0.4127;
        this->add_parameter(
            "primitive jet state",
            jet_state_,
            "Initial primitive state (rho, u, p) for jet state");

        ambient_state_[0] = 5.0;
        ambient_state_[1] = 0.0;
        ambient_state_[2] = 0.4127;
        this->add_parameter(
            "primitive ambient right",
            ambient_state_,
            "Initial primitive state (rho, u, p) for ambient state");

        const auto convert_states = [&]() {
          state_left_ = hyperbolic_system_.from_initial_state(jet_state_);
          state_right_ = hyperbolic_system_.from_initial_state(ambient_state_);
        };
        this->parse_parameters_call_back.connect(convert_states);
        convert_states();
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        return (point[0] < 1.e-12 && std::abs(point[1]) <= jet_width_
                    ? state_left_
                    : state_right_);
      }

    private:
      const HyperbolicSystemView hyperbolic_system_;

      Number gamma_;
      Number jet_width_;

      dealii::Tensor<1, 3, Number> jet_state_;
      dealii::Tensor<1, 3, Number> ambient_state_;

      state_type state_left_;
      state_type state_right_;
    };
  } // namespace EulerInitialStates
} // namespace ryujin
