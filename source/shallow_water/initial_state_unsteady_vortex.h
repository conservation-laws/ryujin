//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <initial_state.h>
#include <hyperbolic_system.h>

namespace ryujin
{
  namespace ShallowWater
  {
    namespace InitialStateLibrary
    {
      /**
       * Eric derived this. Will publish somewhere sometime.
       *
       * @ingroup InitialValues
       */
      template <int dim, typename Number, typename state_type>
      class UnsteadyVortex : public InitialState<dim, Number, state_type, 1>
      {
      public:
        UnsteadyVortex(const HyperbolicSystem &hyperbolic_system,
                       const std::string subsection)
            : InitialState<dim, Number, state_type, 1>("unsteady vortex",
                                                       subsection)
            , hyperbolic_system(hyperbolic_system)
        {
          depth_ = 1.0;
          this->add_parameter(
              "reference depth", depth_, "Reference water depth");

          mach_number_ = 2.0;
          this->add_parameter(
              "mach number", mach_number_, "Mach number of unsteady vortex");

          beta_ = 0.1;
          this->add_parameter("beta", beta_, "vortex strength beta");
        }

        virtual state_type compute(const dealii::Point<dim> &point,
                                   Number t) final override
        {
          const auto gravity = hyperbolic_system.gravity();

          dealii::Point<2> point_bar;
          point_bar[0] = point[0] - mach_number_ * t;
          point_bar[1] = point[1];

          const Number r_square = Number(point_bar.norm_square());

          /* We assume r0 = 1 */
          const Number factor =
              beta_ / Number(2. * M_PI) *
              exp(Number(0.5) - Number(0.5) * r_square /* /r0^ 2*/);

          const Number h = depth_ - Number(1. / (2. * gravity /* * r0^2 */)) *
                                        factor * factor;
          const Number u = mach_number_ - factor * Number(point_bar[1]);
          const Number v = factor * Number(point_bar[0]);


          if constexpr (dim == 2)
            return hyperbolic_system.template expand_state<dim>(
                HyperbolicSystem::state_type<2, Number>{{h, h * u, h * v}});
          else {
            AssertThrow(false, dealii::ExcNotImplemented());
            __builtin_trap();
          }
        }

      private:
        const HyperbolicSystem &hyperbolic_system;

        Number depth_;
        Number mach_number_;
        Number beta_;
      };
    } // namespace InitialStateLibrary
  } // namespace ShallowWater
} // namespace ryujin
