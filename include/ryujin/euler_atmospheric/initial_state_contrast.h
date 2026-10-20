//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// [LANL Copyright Statement]
// Copyright (C) 2022 - 2026 by the ryujin authors
// Copyright (C) 2025 - 2026 by Triad National Security, LLC
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/typed/initial_state.h>

namespace ryujin
{
  namespace EulerAtmosphericInitialStates
  {
    /**
     * An initial state formed by a contrast of a given "left" and "right"
     * primitive (initial) state.
     *
     * @note The @p t argument is ignored. This class does not evolve a
     * possible shock front in time.
     *
     * @ingroup EulerAtmosphericEquations
     */
    template <typename HyperbolicSystem, int dim, typename Number>
    class Contrast : public InitialState<HyperbolicSystem, dim, Number>
    {
    public:
      using View = typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename View::state_type;
      using state_type_1d =
          typename HyperbolicSystem::template View<1, Number>::state_type;

      Contrast(const HyperbolicSystem &hyperbolic_system,
               const std::string subsection)
          : InitialState<HyperbolicSystem, dim, Number>("contrast", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        primitive_left_[0] = 1.4;
        primitive_left_[1] = 0.;
        primitive_left_[2] = 1.;
        this->add_parameter(
            "primitive state left",
            primitive_left_,
            "Initial 1d primitive state (rho, u, theta) on the left");

        primitive_right_[0] = 1.4;
        primitive_right_[1] = 0.;
        primitive_right_[2] = 1.;
        this->add_parameter(
            "primitive state right",
            primitive_right_,
            "Initial 1d primitive state (rho, u, theta) on the right");

        const auto convert_states = [&]() {
          const auto view = hyperbolic_system_.template view<dim, Number>();
          state_left_ = view.from_initial_state(primitive_left_);
          state_right_ = view.from_initial_state(primitive_right_);
        };
        this->parse_parameters_call_back.connect(convert_states);
        convert_states();
      }

      state_type compute(const dealii::Point<dim> &point, Number /*t*/) final
      {
        return (point[0] > 0. ? state_right_ : state_left_);
      }

      auto initial_precomputations(const dealii::Point<dim> &point) ->
          typename InitialState<HyperbolicSystem, dim, Number>::
              initial_precomputed_type final
      {
        /* Return the vertical coordinate: */
        return {Number(point[dim - 1])};
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;

      state_type_1d primitive_left_;
      state_type_1d primitive_right_;

      state_type state_left_;
      state_type state_right_;
    };
  } // namespace EulerAtmosphericInitialStates
} // namespace ryujin
