//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"
#include <initial_state_library.h>

namespace ryujin
{
  namespace Skeleton
  {
    struct Description;

    /**
     * Uniform initial state defined by a given primitive state.
     *
     * @ingroup SkeletonEquations
     */
    template <int dim, typename Number>
    class Uniform : public InitialState<Description, dim, Number>
    {
    public:
      using View = HyperbolicSystemView<dim, Number>;
      using state_type = typename View::state_type;
      using primitive_state_type = typename View::primitive_state_type;

      Uniform(const HyperbolicSystem &hyperbolic_system,
              const std::string subsection)
          : InitialState<Description, dim, Number>("uniform", subsection)
          , hyperbolic_system(hyperbolic_system)
      {
        primitive_[0] = 1.0;
        this->add_parameter(
            "primitive state", primitive_, "Initial 1d primitive state");
      }

      state_type compute(const dealii::Point<dim> & /*point*/,
                         Number /*t*/) final
      {
        const auto view = hyperbolic_system.view<dim, Number>();
        return view.from_primitive_state(view.expand_state(primitive_));
      }

    private:
      const HyperbolicSystem &hyperbolic_system;

      primitive_state_type primitive_;
    };
  } // namespace Skeleton
} // namespace ryujin
