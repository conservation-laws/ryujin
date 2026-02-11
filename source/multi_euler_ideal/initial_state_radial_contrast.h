//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2024 by the ryujin authors
//

#pragma once

#include <initial_state_library.h>

namespace ryujin
{
  namespace MultiSpeciesEulerInitialStates
  {
    /**
     * A modification of the "contrast" initial state. Now, we have an
     * initial state formed by a contrast of a given "left" and "right"
     * primitive state where the "left" state is "inside" the radius, R,
     * and the "right" state is outside.
     *
     * @note The @p t argument is ignored. This class always returns the
     * initial configuration.
     *
     * @ingroup MultiSpeciesEulerEquations
     */
    template <typename Description, int dim, typename Number>
    class RadialContrast : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using View =
          typename Description::template HyperbolicSystemView<dim, Number>;
      using state_type = typename View::state_type;

      RadialContrast(const HyperbolicSystem &hyperbolic_system,
                     const std::string &subsection)
          : InitialState<Description, dim, Number>("radial contrast",
                                                   subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        use_radial_velocity_ = false;
        this->add_parameter(
            "use radial velocity",
            use_radial_velocity_,
            "If set to true, we transform a non-zero velocity into a radial "
            "velocity with scaling 1 / r^(dim - 1)");

        temp_left_[0] = 0.5;
        temp_left_[1] = 1.4;
        temp_left_[2] = 0.0;
        temp_left_[3] = 1.0;
        this->add_parameter(
            "primitive state left",
            temp_left_,
            "Initial 1d primitive state (Y_0, rho, u, p) inside radial area");

        temp_right_[0] = 0.5;
        temp_right_[1] = 1.4;
        temp_right_[2] = 0.0;
        temp_right_[3] = 1.0;
        this->add_parameter(
            "primitive state right",
            temp_right_,
            "Initial 1d primitive state (Y_0, rho, u, p) outside radial area");

        radius_ = 1.0;
        this->add_parameter("radius", radius_, "Radius of radial area");

        const auto convert_states = [&]() {
          const auto view = hyperbolic_system_.template view<dim, Number>();

          const auto primitive_left_ = extend(temp_left_);
          const auto primitive_right_ = extend(temp_right_);

          state_left_ = view.from_initial_state(primitive_left_);
          state_right_ = view.from_initial_state(primitive_right_);
        };

        this->parse_parameters_call_back.connect(convert_states);
        convert_states();
      };

      state_type compute(const dealii::Point<dim> &point, Number /*t*/) final
      {
        if (point.norm() > 0. && use_radial_velocity_) {
          state_left_[2] = temp_left_[2] * point[0] / point.norm();
          state_left_[3] = temp_left_[2] * point[1] / point.norm();

          state_left_[2] = temp_right_[2] * point[0] / point.norm();
          state_left_[3] = temp_right_[2] * point[1] / point.norm();
        }

        auto final_state =
            (point.norm() > radius_ ? state_right_ : state_left_);

        return final_state;
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;

      dealii::Tensor<1, 4, Number> temp_left_;
      dealii::Tensor<1, 4, Number> temp_right_;

      state_type state_left_;
      state_type state_right_;

      double radius_;
      bool use_radial_velocity_;


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
