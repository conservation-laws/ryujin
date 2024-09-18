//
// SPDX-License-Identifier: Apache-2.0
// [LANL Copyright Statement]
// Copyright (C) 2022 - 2024 by the ryujin authors
// Copyright (C) 2023 - 2024 by Triad National Security, LLC
//

#pragma once

#include <initial_state_library.h>

namespace ryujin
{
  namespace ShallowWaterInitialStates
  {
    /**
     * Solitary wave over flat bottom.
     *
     * @ingroup ShallowWaterEquations
     */
    template <typename Description, int dim, typename Number>
    class Soliton : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using View =
          typename Description::template HyperbolicSystemView<dim, Number>;
      using state_type = typename View::state_type;

      Soliton(const HyperbolicSystem &hyperbolic_system,
              const std::string subsection)
          : InitialState<Description, dim, Number>("soliton", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        depth_ = 1;
        this->add_parameter(
            "still water depth", depth_, "Depth of still water");

        amplitude_ = 0.1;
        this->add_parameter("amplitude", amplitude_, "Amplitude of soliton");
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        const auto view = hyperbolic_system_.template view<dim, Number>();
        const Number g = view.gravity();

        const auto &x = point[0];

        const Number celerity = std::sqrt(g * (amplitude_ + depth_));
        const Number width = std::sqrt(
            3. * amplitude_ / (4. * depth_ * depth_ * (amplitude_ + depth_)));
        const Number sechSqd =
            1. / std::pow(cosh(width * (x - celerity * t)), 2);

        /* If there is bathymetry, take max of profile and 0 */
        const Number profile = depth_ + amplitude_ * sechSqd;
        const Number h = std::max(profile, Number(0.));

        const Number v = celerity * (profile - depth_) / profile;

        return state_type{{h, h * v}};
      }

      /* Default bathymetry of 0 */

    private:
      const HyperbolicSystem &hyperbolic_system_;

      Number depth_;
      Number amplitude_;
    };

  } // namespace ShallowWaterInitialStates
} // namespace ryujin
