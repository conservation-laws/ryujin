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
    /**
     * Dam break with three conical islands as obstacles. See Sec.~7.8 in
     * @cite GuermondEtAl2018SW with \f$D = [0,75m]x[0,30m]\f$.
     *
     * @ingroup InitialValues
     */
    template <int dim, typename Number, typename state_type>
    class ThreeBumpsDamBreak : public InitialState<dim, Number, state_type, 1>
    {
    public:
      ThreeBumpsDamBreak(const HyperbolicSystem &hyperbolic_system,
                         const std::string s)
          : InitialState<dim, Number, state_type, 1>("three bumps dam break", s)
          , hyperbolic_system(hyperbolic_system)
      {
        well_balancing_validation = false;
        this->add_parameter(
            "well balancing validation",
            well_balancing_validation,
            "If set to true then the initial profile is returned for all "
            "times "
            "(t>0); otherwise a constant inflow is computed for t>0 suitable "
            "for prescribing Dirichlet conditions at the inflow boundary.");


        left_depth = 1.875;
        this->add_parameter("left water depth",
                            left_depth,
                            "Depth of water to the left of pseudo-dam");
        right_depth = 0.;
        this->add_parameter("right water depth",
                            right_depth,
                            "Depth of water to the right of pseudo-dam");

        cone_magnitude = 1.;
        this->add_parameter("cone magnitude",
                            cone_magnitude,
                            "To modify magnitude of cone heights");
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        const Number x = point[0];

        /* Initial state: */

        if (t <= 1.e-10 || well_balancing_validation) {
          Number h = x < 0 ? left_depth : right_depth;
          h = std::max(h - compute_bathymetry(point), Number(0.));
          return hyperbolic_system.template expand_state<dim>(
              HyperbolicSystem::state_type<1, Number>{{h, Number(0.)}});
        }

        /* For t > 0 prescribe constant inflow Dirichlet data on the left: */

        const auto &h = left_depth;
        const auto a = hyperbolic_system.speed_of_sound(
            HyperbolicSystem ::state_type<1, Number>{{h, Number(0.)}});
        return hyperbolic_system.template expand_state<dim>(
            HyperbolicSystem::state_type<1, Number>{{h, h * a}});
      }

      auto initial_precomputations(const dealii::Point<dim> &point) ->
          typename InitialState<dim, Number, state_type, 1>::precomputed_type
          final
      {
        /* Compute bathymetry: */
        return {compute_bathymetry(point)};
      }

    private:
      const HyperbolicSystem &hyperbolic_system;

      DEAL_II_ALWAYS_INLINE inline Number
      compute_bathymetry(const dealii::Point<dim> &point) const
      {
        if constexpr (dim == 1) {
          /* When dim = 1, we only have one cone */
          const Number &x = point[0];

          Number z3 = 3. - 3. / 10. * std::sqrt(std::pow(x - 47.5, 2));
          return cone_magnitude * std::max({z3, Number(0.)});
        }

        const Number &x = point[0];
        const Number &y = point[1];

        Number z1 =
            1. -
            1. / 8. * std::sqrt(std::pow(x - 30., 2) + std::pow(y - 6., 2));

        Number z2 =
            1. -
            1. / 8. * std::sqrt(std::pow(x - 30., 2) + std::pow(y - 24., 2));

        Number z3 =
            3. -
            3. / 10. * std::sqrt(std::pow(x - 47.5, 2) + std::pow(y - 15., 2));

        return cone_magnitude * std::max({z1, z2, z3, Number(0.)});
      }

      bool well_balancing_validation;
      Number left_depth;
      Number right_depth;
      Number cone_magnitude;
    };

  } // namespace ShallowWater
} // namespace ryujin
