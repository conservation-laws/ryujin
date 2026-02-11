//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2024 by the ryujin authors
// Copyright (C) 2025 by Triad National Security, LLC
//

#pragma once

#include <initial_state_library.h>

namespace ryujin
{
  namespace MultiSpeciesEulerInitialStates
  {
    /**
     * An initial state formed by a contrast of a given "left" and "right"
     * primitive (initial) state.
     *
     * @note The @p t argument is ignored.
     *
     * @ingroup MultiSpeciesEulerEquations
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
        temp_left_[0] = 0.5;
        temp_left_[1] = 1.4;
        temp_left_[2] = 0.0;
        temp_left_[3] = 1.0;
        this->add_parameter(
            "primitive state left",
            temp_left_,
            "Initial 1d primitive state (Y_0, rho, u, p) on the left");

        temp_right_[0] = 0.5;
        temp_right_[1] = 1.4;
        temp_right_[2] = 0.0;
        temp_right_[3] = 1.0;
        this->add_parameter(
            "primitive state right",
            temp_right_,
            "Initial 1d primitive state (Y_0, rho, u, p) on the right");

        const auto convert_states = [&]() {
          const auto view = hyperbolic_system_.template view<dim, Number>();

          const auto primitive_left_ = extend(temp_left_);
          const auto primitive_right_ = extend(temp_right_);

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

      dealii::Tensor<1, 4, Number> temp_left_;
      dealii::Tensor<1, 4, Number> temp_right_;

      state_type state_left_;
      state_type state_right_;

      DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, 4, Number>
      extend(dealii::Tensor<1, 4, Number> &temp_in) const
      {
        dealii::Tensor<1, 4, Number> result;
        result[0] = temp_in[0] * temp_in[1];        // = alpha_0 rho_0;
        result[1] = (1. - temp_in[0]) * temp_in[1]; // = alpha_1 rho_1;

        for (unsigned int i = 2; i < 4; ++i)
          result[i] = temp_in[i];

        return result;
      }
    };
  } // namespace MultiSpeciesEulerInitialStates
} // namespace ryujin
