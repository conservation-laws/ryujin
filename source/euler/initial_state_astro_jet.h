//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

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
      using View = typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename View::state_type;
      using state_type_1d =
          typename HyperbolicSystem::template View<1, Number>::state_type;

      AstroJet(const HyperbolicSystem &hyperbolic_system,
               const std::string subsection)
          : InitialState<Description, dim, Number>("astro jet", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        jet_width_ = 0.05;
        this->add_parameter("jet width",
                            jet_width_,
                            "The width of the jet coming out of boundary");

        jet_state_[0] = 5.0;
        jet_state_[1] = 30.0;
        if constexpr (View::have_energy_equation)
          jet_state_[2] = 0.4127;
        this->add_parameter(
            "primitive jet state",
            jet_state_,
            "1d primitive state [rho, u, p] for jet state (or [rho, u] for "
            "the barotropic Euler module)");

        ambient_state_[0] = 5.0;
        ambient_state_[1] = 0.0;
        if constexpr (View::have_energy_equation)
          ambient_state_[2] = 0.4127;
        this->add_parameter(
            "primitive ambient right",
            ambient_state_,
            "1d primitive state [rho, u, p] for ambient state (or [rho, u] for "
            "the barotropic Euler module)");

        const auto convert_states = [&]() {
          const auto view = hyperbolic_system_.template view<dim, Number>();
          state_left_ = view.from_initial_state(jet_state_);
          state_right_ = view.from_initial_state(ambient_state_);
        };
        this->parse_parameters_call_back.connect(convert_states);
        convert_states();
      }

      state_type compute(const dealii::Point<dim> &point, Number /*t*/) final
      {
        return (point[0] < 1.e-12 && std::abs(point[1]) <= jet_width_
                    ? state_left_
                    : state_right_);
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;

      Number jet_width_;

      state_type_1d jet_state_;
      state_type_1d ambient_state_;

      state_type state_left_;
      state_type state_right_;
    };
  } // namespace EulerInitialStates
} // namespace ryujin
