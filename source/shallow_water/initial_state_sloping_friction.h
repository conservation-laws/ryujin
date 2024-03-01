//
// SPDX-License-Identifier: BSD-3-Clause
// [LANL Copyright Statement]
// Copyright (C) 2024 by the ryujin authors
// Copyright (C) 2023 - 2024 by Triad National Security, LLC
//

#pragma once

#include <initial_state_library.h>

#include <simd.h>

namespace ryujin
{
  namespace ShallowWaterInitialStates
  {
    /**
     * A 1D benchmark configuration consisting of a steady flow over an inclined
     * plane with Manning's friction. See section 4.1 of @Chertock2015 for
     * details.
     *
     * @ingroup ShallowWaterEquations
     */
    template <typename Description, int dim, typename Number>
    class SlopingFriction : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using View =
          typename Description::template HyperbolicSystemView<dim, Number>;
      using state_type = typename View::state_type;

      SlopingFriction(const HyperbolicSystem &hyperbolic_system,
                      const std::string subsection)
          : InitialState<Description, dim, Number>("sloping friction",
                                                   subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        slope_ = 1.;
        this->add_parameter(
            "ramp slope",
            slope_,
            "The (positive) slope of the inclined plane used to "
            "define the bathymetry");

        q_0_ = 0.1;
        this->add_parameter("initial discharge",
                            q_0_,
                            "The initial (unit) discharge in [m^2 / s]");
      }

      state_type compute(const dealii::Point<dim> & /*point*/,
                         Number /*t*/) final
      {
        const auto view = hyperbolic_system_.template view<dim, Number>();

        /*
         * Water depth profile depends on slope, discharge and Manning's
         * coefficient. The gamma quantity in ref is fixed to 4. / 3.
         */

        const Number n = view.manning_friction_coefficient();
        const Number exponent = 1. / (2. + 4. / 3.);

        Number profile = n * n * q_0_ * q_0_ / slope_;
        Number h = ryujin::pow(profile, exponent);

        return state_type{{h, q_0_}};
      }

      auto initial_precomputations(const dealii::Point<dim> &point) ->
          typename InitialState<Description, dim, Number>::
              precomputed_state_type final
      {
        /* Compute bathymetry: */
        return {compute_bathymetry(point)};
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;

      DEAL_II_ALWAYS_INLINE inline Number
      compute_bathymetry(const dealii::Point<dim> &point) const
      {
        return -slope_ * point[0];
      }

      Number slope_;
      Number q_0_;
    };

  } // namespace ShallowWaterInitialStates
} // namespace ryujin
