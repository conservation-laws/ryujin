//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"
#include <initial_state_library.h>

namespace ryujin
{
  namespace ShallowWater
  {
    struct Description;

    /**
     * A 1D/2D Paraboloid configuration. See following reference:
     *
     * 2008 paper
     *
     * @ingroup ShallowWaterEquations
     */
    template <int dim, typename Number>
    class NewParaboloid : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystemView = HyperbolicSystem::View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;
      using primitive_state_type =
          typename HyperbolicSystemView::primitive_state_type;

      NewParaboloid(const HyperbolicSystem &hyperbolic_system,
                    const std::string subsection)
          : InitialState<Description, dim, Number>("new paraboloid", subsection)
          , hyperbolic_system(hyperbolic_system)
      {
        a_ = 3000.;
        this->add_parameter(
            "free surface radius", a_, "Radius of the circular free surface");

        length_ = 10000.;
        this->add_parameter(
            "paraboloid length", length_, "Length of paraboloid");

        h_0_ = 10.;
        this->add_parameter(
            "water height", h_0_, "Water height at central point");

        B_ = 2.;
        this->add_parameter("speed", B_, "The paraboloid speed");
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        /* Initialize primitive variables */
        Number h = 0.;   // water depth
        Number v_x = 0.; // velocity in x-direction
        Number v_y = 0.; // velocity in y-direction (if in 2D)

        /* Common quantities */
        const auto g = hyperbolic_system.gravity();
        const Number k = 0.;
        const Number p = std::sqrt(8. * g * h_0_) / a_;
        const Number s = std::sqrt(p * p - k * k) / 2.;


        /* Define profiles for each dimension */
        switch (dim) {
        case 1:
          /* Fake 1D configuration */
          {
            auto term1 =
                (a_ * a_ * B_ * B_) / (8. * g * g * h_0_) * std::exp(-k * t);
            term1 *= (1. / 4. * k * k - s * s) * std::cos(2. * s * t) -
                     s * k * sin(2. * s * t);

            const auto term2 = -(B_ * B_ / (4. * g)) * std::exp(-k * t);

            auto term3 = -(B_ / g) * std::exp(-1. / 2. * k * t);
            term3 *= (s * std::cos(s * t) + 1. / 2. * k * std::sin(s * t)) *
                     (point[0] - 1. / 2. * length_);

            auto htilde = h_0_ - compute_bathymetry(point);
            htilde += term1 + term2 + term3;

            h = std::max(htilde, 0.);

            v_x = B_ * std::exp(-1. / 2. * k * t) * std::sin(s * t);
          }
          break;

        default:
          AssertThrow(false, dealii::ExcNotImplemented());
          __builtin_trap();
        }

        /* Set final state */
        if constexpr (dim == 1)
          return state_type{{h, h * v_x}};

        else if constexpr (dim == 2)
          return state_type{{h, h * v_x, h * v_y}};

        else {
          AssertThrow(false, dealii::ExcNotImplemented());
          __builtin_trap();
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

      DEAL_II_ALWAYS_INLINE inline Number
      compute_bathymetry(const dealii::Point<dim> &point) const
      {
        return h_0_ / (a_ * a_) * std::pow(point[0] - 0.5 * length_, 2);
      }

      Number a_;
      Number h_0_;
      Number B_;
      Number length_;
    };

  } // namespace ShallowWater
} // namespace ryujin
