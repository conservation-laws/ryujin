//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"
#include <initial_state.h>

namespace ryujin
{
  namespace Euler
  {
    /**
     * Uniform initial state defined by a given primitive state.
     *
     * @ingroup EulerEquations
     */
    template <int dim, typename Number, typename state_type>
    class Uniform : public InitialState<dim, Number, state_type>
    {
    public:
      using HyperbolicSystemView = HyperbolicSystem::View<dim, Number>;

      Uniform(const HyperbolicSystemView &hyperbolic_system,
              const std::string subsection)
          : InitialState<dim, Number, state_type>("uniform", subsection)
          , hyperbolic_system(hyperbolic_system)
      {
        primitive_[0] = hyperbolic_system.gamma();
        primitive_[1] = 3.0;
        primitive_[2] = 1.;
        this->add_parameter("primitive state",
                            primitive_,
                            "Initial 1d primitive state (rho, u, p)");
      }

      state_type compute(const dealii::Point<dim> & /*point*/,
                         Number /*t*/) final
      {
        return hyperbolic_system.from_primitive_state(
            hyperbolic_system.expand_state(primitive_));
      }

    private:
      const HyperbolicSystemView &hyperbolic_system;

      dealii::Tensor<1, 3, Number> primitive_;
    };
  } // namespace Euler
} // namespace ryujin
