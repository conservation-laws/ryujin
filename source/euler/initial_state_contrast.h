//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2024 by the ryujin authors
//

#pragma once

#include <initial_state_library.h>

namespace ryujin
{
  namespace EulerInitialStates
  {
    /**
     * An initial state formed by a contrast of a given "left" and "right"
     * primitive (initial) state.
     *
     * @note The @p t argument is ignored. This class does not evolve a
     * possible shock front in time. If you need correct time-dependent
     * Dirichlet data use @ref ShockFront instead.
     *
     * @ingroup EulerEquations
     */
    template <typename Description, int dim, typename Number>
    class Contrast : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using View =
          typename Description::template HyperbolicSystemView<dim, Number>;
      using state_type = typename View::state_type;

      Contrast(const HyperbolicSystem &hyperbolic_system,
               const std::string subsection)
          : InitialState<Description, dim, Number>("contrast", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        primitive_left_[0] = 1.4;
        primitive_left_[1] = 0.;
        primitive_left_[2] = 1.;
        this->add_parameter(
            "primitive state left",
            primitive_left_,
            "Initial 1d primitive state (rho, u, p) on the left");

        primitive_right_[0] = 1.4;
        primitive_right_[1] = 0.;
        primitive_right_[2] = 1.;
        this->add_parameter(
            "primitive state right",
            primitive_right_,
            "Initial 1d primitive state (rho, u, p) on the right");

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

    private:
      const HyperbolicSystem &hyperbolic_system_;

      dealii::Tensor<1, 3, Number> primitive_left_;
      dealii::Tensor<1, 3, Number> primitive_right_;

      state_type state_left_;
      state_type state_right_;
    };
  } // namespace EulerInitialStates
} // namespace ryujin
