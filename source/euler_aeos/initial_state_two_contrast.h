//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <hyperbolic_system.h>
#include <initial_state.h>

namespace ryujin
{
  namespace EulerAEOS
  {
    namespace InitialStateLibrary
    {
      /**
       *
       * An initial state formed by two contrasts of given "left", "middle"
       * and "right" primitive states. The user defines the lengths of the left
       * and middle regions. The rest of the domain is populated with the right
       * region. This initial state (default values) can be used to replicate
       * the classical Woodward-Colella colliding blast wave problem described
       * in:
       *
       * Woodward, P. and Colella, P., "The Numerical Simulation of
       * Two-Dimensional Fluid Flow with Strong Shocks", J. Computational
       * Physics, 54, 115-173 (1984).
       *
       * @ingroup InitialValues
       */
      template <int dim, typename Number, typename state_type>
      class TwoContrast : public InitialState<dim, Number, state_type>
      {
      public:
        TwoContrast(const HyperbolicSystem &hyperbolic_system,
                    const std::string subsection)
            : InitialState<dim, Number, state_type>("TwoContrast", subsection)
            , hyperbolic_system(hyperbolic_system)
        {
          primitive_left_[0] = 1.;
          primitive_left_[1] = 0.;
          primitive_left_[2] = 1.e3 / (1.4 - 1) / 1.; // e = p / (gam - 1) / rho
          this->add_parameter(
              "primitive state left",
              primitive_left_,
              "Initial 1d primitive state (rho, u, e) on the left");

          left_length_ = 0.1;
          this->add_parameter("left region length",
                              left_length_,
                              "The length of the left region");

          primitive_middle_[0] = 1.;
          primitive_middle_[1] = 0.;
          primitive_middle_[2] = 1.e-2 / (1.4 - 1) / 1.;
          this->add_parameter(
              "primitive state middle",
              primitive_middle_,
              "Initial 1d primitive state (rho, u, e) in the middle");

          middle_length_ = 0.8;
          this->add_parameter("middle region length",
                              middle_length_,
                              "The length of the middle region");

          primitive_right_[0] = 1.;
          primitive_right_[1] = 0.;
          primitive_right_[2] = 1.e2 / (1.4 - 1) / 1.;
          this->add_parameter(
              "primitive state right",
              primitive_right_,
              "Initial 1d primitive state (rho, u, e) on the right");
        }

        state_type compute(const dealii::Point<dim> &point, Number /*t*/) final
        {
          auto temp = hyperbolic_system.from_primitive_state(
              point[0] >= left_length_ ? primitive_middle_ : primitive_left_);

          if (point[0] >= left_length_ + middle_length_)
            temp = hyperbolic_system.from_primitive_state(primitive_right_);

          return hyperbolic_system.template expand_state<dim>(temp);
        }

      private:
        const HyperbolicSystem &hyperbolic_system;

        Number left_length_;
        Number middle_length_;

        dealii::Tensor<1, 3, Number> primitive_left_;
        dealii::Tensor<1, 3, Number> primitive_middle_;
        dealii::Tensor<1, 3, Number> primitive_right_;
      };
    } // namespace InitialStateLibrary
  }   // namespace EulerAEOS
} // namespace ryujin
