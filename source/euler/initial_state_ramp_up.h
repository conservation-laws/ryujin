//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <initial_state_library.h>

namespace ryujin
{
  namespace EulerInitialStates
  {
    /**
     * A time-dependent state given by an initial state @p primite_left_
     * valid for \f$ t \le t_{\text{left}} \f$ and a final state @p
     * primite_right_ valid for \f$ t \ge t_{\text{right}} \f$. In between,
     * a smooth interpolation is performed.
     *
     * @ingroup EulerEquations
     */
    template <typename Description, int dim, typename Number>
    class RampUp : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using HyperbolicSystemView =
          typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      RampUp(const HyperbolicSystemView &hyperbolic_system,
             const std::string subsection)
          : InitialState<Description, dim, Number>("ramp up", subsection)
          , hyperbolic_system(hyperbolic_system)
      {
        primitive_initial_[0] = 1.4;
        primitive_initial_[1] = 0.0;
        primitive_initial_[2] = 1.;
        this->add_parameter("primitive state initial",
                            primitive_initial_,
                            "Initial 1d primitive state (rho, u, p)");

        primitive_final_[0] = 1.4;
        primitive_final_[1] = 3.0;
        primitive_final_[2] = 1.;
        this->add_parameter("primitive state final",
                            primitive_final_,
                            "Final 1d primitive state (rho, u, p)");

        t_initial_ = 0.;
        this->add_parameter("time initial",
                            t_initial_,
                            "Time until which initial state is prescribed");

        t_final_ = 1.;
        this->add_parameter("time final",
                            t_final_,
                            "Time from which on the final state is attained)");

        const auto convert_states = [&]() {
          state_initial_ =
              hyperbolic_system.from_initial_state(primitive_initial_);
          state_final_ = hyperbolic_system.from_initial_state(primitive_final_);
        };
        this->parse_parameters_call_back.connect(convert_states);
        convert_states();
      }

      auto compute(const dealii::Point<dim> & /*point*/, Number t)
          -> state_type final
      {
        state_type result;

        if (t <= t_initial_) {
          result = state_initial_;
        } else if (t >= t_final_) {
          result = state_final_;
        } else {
          const Number factor =
              std::cos(0.5 * M_PI * (t - t_initial_) / (t_final_ - t_initial_));

          const Number alpha = factor * factor;
          const Number beta = Number(1.) - alpha;
          result = alpha * state_initial_ + beta * state_final_;
        }

        return result;
      }

    private:
      const HyperbolicSystemView hyperbolic_system;

      dealii::Tensor<1, 3, Number> primitive_initial_;
      dealii::Tensor<1, 3, Number> primitive_final_;
      Number t_initial_;
      Number t_final_;

      state_type state_initial_;
      state_type state_final_;
    };
  } // namespace Euler
} // namespace ryujin
