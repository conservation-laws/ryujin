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
     * Returns a uniform initial state defined by a given primitive
     * (initial) state.
     *
     * @note The @p t argument is ignored. This class always returns the
     * initial configuration.
     *
     * @ingroup EulerEquations
     */
    template <typename Description, int dim, typename Number>
    class Uniform : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using HyperbolicSystemView =
          typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      Uniform(const HyperbolicSystem &hyperbolic_system,
              const std::string subsection)
          : InitialState<Description, dim, Number>("uniform", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        primitive_[0] = 1.4;
        primitive_[1] = 3.;
        primitive_[2] = 1.;
        this->add_parameter("primitive state",
                            primitive_,
                            "Initial 1d primitive state (rho, u, p)");

        const auto convert_states = [&]() {
          state_ = hyperbolic_system_.from_initial_state(primitive_);
        };
        this->parse_parameters_call_back.connect(convert_states);
        convert_states();
      }

      state_type compute(const dealii::Point<dim> & /*point*/,
                         Number /*t*/) final
      {
        return state_;
      }

    private:
      const HyperbolicSystemView hyperbolic_system_;

      dealii::Tensor<1, 3, Number> primitive_;

      state_type state_;
    };
  } // namespace EulerInitialStates
} // namespace ryujin
