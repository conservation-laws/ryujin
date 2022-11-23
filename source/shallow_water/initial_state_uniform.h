//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <hyperbolic_system.h>
#include <initial_state.h>

namespace ryujin
{
  namespace ShallowWater
  {
    namespace InitialStateLibrary
    {
      /**
       * Uniform initial state defined by a given primitive state.
       *
       * @ingroup InitialValues
       */
      template <int dim, typename Number, typename state_type>
      class Uniform : public InitialState<dim, Number, state_type, 1>
      {
      public:
        Uniform(const HyperbolicSystem &hyperbolic_system,
                const std::string subsection)
            : InitialState<dim, Number, state_type, 1>("uniform", subsection)
            , hyperbolic_system(hyperbolic_system)
        {
          primitive_[0] = 1.;
          primitive_[1] = 5.0;
          this->add_parameter("primitive state",
                              primitive_,
                              "Initial 1d primitive state (h, u)");
        }

        state_type compute(const dealii::Point<dim> & /*point*/,
                           Number /*t*/) final
        {
          const auto temp = hyperbolic_system.from_primitive_state(primitive_);
          return hyperbolic_system.template expand_state<dim>(temp);
        }

        /* Default bathymetry of 0 */

      private:
        const HyperbolicSystem &hyperbolic_system;

        dealii::Tensor<1, 2, Number> primitive_;
      };
    } // namespace InitialStateLibrary
  }   // namespace ShallowWater
} // namespace ryujin
