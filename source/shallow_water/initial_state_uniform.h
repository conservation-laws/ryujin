//
// SPDX-License-Identifier: MIT or BSD-3-Clause
// [LANL Copyright Statement]
// Copyright (C) 2020 - 2023 by the ryujin authors
// Copyright (C) 2023 - 2023 by Triad National Security, LLC
//

#pragma once

#include <initial_state_library.h>

namespace ryujin
{
  namespace ShallowWaterInitialStates
  {
    /**
     * Returns a uniform initial state defined by a given primitive
     * (initial) state.
     *
     * @note The @p t argument is ignored. This class always returns the
     * initial configuration.
     *
     * @ingroup ShallowWaterEquations
     */
    template <typename Description, int dim, typename Number>
    class Uniform : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using HyperbolicSystemView =
          typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      Uniform(const HyperbolicSystem &hyperbolic_system,
              const std::string subsection)
          : InitialState<Description, dim, Number>("uniform", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        primitive_[0] = 1.;
        primitive_[1] = 1.;
        this->add_parameter(
            "primitive state", primitive_, "Initial 1d primitive state (h, u)");
      }

      state_type compute(const dealii::Point<dim> & /*point*/,
                         Number /*t*/) final
      {
        return hyperbolic_system_.from_initial_state(primitive_);
      }

      /* Default bathymetry of 0 */

    private:
      const HyperbolicSystemView hyperbolic_system_;

      dealii::Tensor<1, 2, Number> primitive_;
    };

  } // namespace ShallowWaterInitialStates
} // namespace ryujin
