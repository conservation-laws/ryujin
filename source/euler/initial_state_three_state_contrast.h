//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include <initial_state_library.h>

namespace ryujin
{
  namespace EulerInitialStates
  {
    /**
     * An initial state formed by two contrasts of given "left", "middle"
     * and "right" primitive states. The user defines the lengths of the left
     * and middle regions. The rest of the domain is populated with the right
     * region. This initial state (default values) can be used to replicate
     * the classical Woodward-Colella colliding blast wave problem described
     * in @cite Woodward1984
     *
     * @note The @p t argument is ignored. This class always returns the
     * initial configuration.
     *
     * @ingroup EulerEquations
     */
    template <typename Description, int dim, typename Number>
    class ThreeStateContrast : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using View =
          typename Description::template HyperbolicSystemView<dim, Number>;
      using state_type = typename View::state_type;

      ThreeStateContrast(const HyperbolicSystem &hyperbolic_system,
                         const std::string &subsection)
          : InitialState<Description, dim, Number>("three state contrast",
                                                   subsection)
          , hyperbolic_system_(hyperbolic_system)
      {

        primitive_left_[0] = 1.;
        primitive_left_[1] = 0.;
        primitive_left_[2] = 1.e3;
        this->add_parameter(
            "primitive state left",
            primitive_left_,
            "Initial 1d primitive state (rho, u, p) on the left");

        left_length_ = 0.1;
        this->add_parameter("left region length",
                            left_length_,
                            "The length of the left region");

        primitive_middle_[0] = 1.;
        primitive_middle_[1] = 0.;
        primitive_middle_[2] = 1.e-2;
        this->add_parameter(
            "primitive state middle",
            primitive_middle_,
            "Initial 1d primitive state (rho, u, p) in the middle");

        middle_length_ = 0.8;
        this->add_parameter("middle region length",
                            middle_length_,
                            "The length of the middle region");

        primitive_right_[0] = 1.;
        primitive_right_[1] = 0.;
        primitive_right_[2] = 1.e2;
        this->add_parameter(
            "primitive state right",
            primitive_right_,
            "Initial 1d primitive state (rho, u, p) on the right");

        const auto convert_states = [&]() {
          const auto view = hyperbolic_system_.template view<dim, Number>();
          state_left_ = view.from_initial_state(primitive_left_);
          state_middle_ = view.from_initial_state(primitive_middle_);
          state_right_ = view.from_initial_state(primitive_right_);
        };
        this->parse_parameters_call_back.connect(convert_states);
        convert_states();
      }

      state_type compute(const dealii::Point<dim> &point, Number /*t*/) final
      {
        return point[0] >= left_length_ + middle_length_ ? state_right_
               : point[0] >= left_length_                ? state_middle_
                                                         : state_left_;
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;

      Number left_length_;
      Number middle_length_;

      dealii::Tensor<1, 3, Number> primitive_left_;
      dealii::Tensor<1, 3, Number> primitive_middle_;
      dealii::Tensor<1, 3, Number> primitive_right_;

      state_type state_left_;
      state_type state_middle_;
      state_type state_right_;
    };
  } // namespace EulerInitialStates
} // namespace ryujin
