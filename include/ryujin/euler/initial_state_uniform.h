//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/typed/initial_state.h>

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
    template <typename HyperbolicSystem, int dim, typename Number>
    class Uniform : public InitialState<HyperbolicSystem, dim, Number>
    {
    public:
      using View = typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename View::state_type;
      using state_type_1d =
          typename HyperbolicSystem::template View<1, Number>::state_type;

      Uniform(const HyperbolicSystem &hyperbolic_system,
              const std::string subsection)
          : InitialState<HyperbolicSystem, dim, Number>("uniform", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        primitive_[0] = 1.4;
        primitive_[1] = 3.;
        if constexpr (View::have_energy_equation)
          primitive_[2] = 1.;
        this->add_parameter("primitive state",
                            primitive_,
                            "1d primitive state [rho, u, p] (or [rho, u] for "
                            "the barotropic Euler module)");

        const auto convert_states = [&]() {
          const auto view = hyperbolic_system_.template view<dim, Number>();
          state_ = view.from_initial_state(primitive_);
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
      const HyperbolicSystem &hyperbolic_system_;

      state_type_1d primitive_;

      state_type state_;
    };
  } // namespace EulerInitialStates
} // namespace ryujin
