//
// SPDX-License-Identifier: MIT or BSD-3-Clause
// [LANL Copyright Statement]
// Copyright (C) 2020 - 2023 by the ryujin authors
// Copyright (C) 2023 - 2023 by Triad National Security, LLC
//

#pragma once

#include <initial_state_library.h>

#include <deal.II/base/function_parser.h>

namespace ryujin
{
  namespace ShallowWaterInitialStates
  {
    /**
     * Wetting/drying test proposed in Section 3.8 of @hou2013.
     *
     * @ingroup ShallowWaterEquations
     */
    template <typename Description, int dim, typename Number>
    class HouTest : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using HyperbolicSystemView =
          typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      HouTest(const HyperbolicSystem &hyperbolic_system, const std::string s)
          : InitialState<Description, dim, Number>("hou test", s)
          , hyperbolic_system(hyperbolic_system)
      {
        depth_ = 35;
        this->add_parameter("reservoir water depth",
                            depth_,
                            "Depth of water in reservoir behind dam");
      }

      state_type compute(const dealii::Point<dim> &point, Number /*t*/) final
      {
        if constexpr (dim == 1) {
          AssertThrow(false, dealii::ExcNotImplemented());
          __builtin_trap();

        } else {
          const Number x = point[0];

          const Number bath = compute_bathymetry(point);

          /* Set water depth behind resevoir */
          Number h = 0.;
          if (x < -100.)
            h = std::max(depth_ - bath, Number(0.));

          return state_type{{h, 0.}};
        }
      }

      auto initial_precomputations(const dealii::Point<dim> &point) ->
          typename InitialState<Description, dim, Number>::
              precomputed_state_type final
      {
        /* Compute bathymetry: */
        return {compute_bathymetry(point)};
      }

    private:
      const HyperbolicSystemView hyperbolic_system;
      Number depth_;

      DEAL_II_ALWAYS_INLINE inline Number
      compute_bathymetry(const dealii::Point<dim> &point) const
      {
        const Number x = point[0];
        const Number y = point[1];

        Number base;
        {
          Number base1 = std::pow(x + 250., 2) / 1600. + std::pow(y, 2) / 400.;
          Number base2 = std::pow(x, 2) / 225. + std::pow(y - 50., 2) / 225.;
          Number base3 =
              std::pow(x - 250., 2) / 1225. + std::pow(y, 2) / 225. - 10.;

          base = std::min(base1, base2);
          base = std::min(base, base3);
        }

        Number bumps;
        {
          Number bump1 =
              80. - std::pow(x + 250., 2) / 50. - std::pow(y, 2) / 50.;

          Number bump2 = (std::pow(x - 200., 2) + std::pow(y + 10., 2) <= 1000.)
                             ? 10.
                             : 0.;
          Number bump3 = (std::abs(x - 380.) <= 40. && std::abs(y - 50.) <= 40.)
                             ? 20.
                             : 0.;

          bumps = std::max(bump1, bump2);
          bumps = std::max(bumps, bump3);
        }
        return std::max(base, bumps);
      }
    };

  } // namespace ShallowWaterInitialStates
} // namespace ryujin
