//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <initial_state_library.h>

namespace ryujin
{
  namespace EulerInitialStates
  {
    /**
     * Uniform initial state defined by a given primitive state.
     *
     * @ingroup EulerEquations
     */
    template <typename Description, int dim, typename Number>
    class Uniform : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using HyperbolicSystemView =
          typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      Uniform(const HyperbolicSystemView &hyperbolic_system,
              const std::string subsection)
          : InitialState<Description, dim, Number>("uniform", subsection)
          , hyperbolic_system(hyperbolic_system)
      {
        primitive_[0] = 1.4;
        primitive_[1] = 3.;
        primitive_[2] = 1.;
        this->add_parameter("primitive state",
                            primitive_,
                            "Initial 1d primitive state (rho, u, p)");

        // FIXME: update primitive
      }

      state_type compute(const dealii::Point<dim> & /*point*/,
                         Number /*t*/) final
      {
        return hyperbolic_system.from_primitive_state(
            hyperbolic_system.expand_state(primitive_));
      }

    private:
      const HyperbolicSystemView hyperbolic_system;

      dealii::Tensor<1, 3, Number> primitive_;
    };
  } // namespace Euler
} // namespace ryujin
