//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include <initial_state_library.h>

namespace ryujin
{
  namespace ShallowWaterInitialStates
  {
    /**
     * A 1D/2D configuration for planar surface flow in a radially-symmetric
     * paraboloid basin without friction. See Section 4.2.2 of
     * @cite swashes_2013 for details.
     *
     * @todo Add variation with friction.
     *
     * @ingroup ShallowWaterEquations
     */
    template <typename Description, int dim, typename Number>
    class Paraboloid : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using HyperbolicSystemView =
          typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      Paraboloid(const HyperbolicSystem &hyperbolic_system,
                 const std::string subsection)
          : InitialState<Description, dim, Number>("paraboloid", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        a_ = 1.;
        this->add_parameter(
            "free surface radius", a_, "Radius of the circular free surface");

        h_0_ = 0.1;
        this->add_parameter(
            "water height", h_0_, "Water height at central point");

        eta_ = 0.5;
        this->add_parameter("eta", eta_, "The eta parameter");
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        /* Common quantities */
        const auto z = compute_bathymetry(point);
        const auto g = hyperbolic_system_.gravity();
        const Number omega = std::sqrt(2. * g * h_0_) / a_;
        const Number &x = point[0];

        /* Initialize primitive variables */
        Number h, v_x, v_y = 0.;

        /* Define profiles for each dimension */
        if constexpr (dim == 1) {

          const auto elevation =
              eta_ * h_0_ / (a_ * a_) * (2. * x * std::cos(omega * t));

          h = std::max(elevation - z, Number(0.));
          v_x = -eta_ * omega * std::sin(omega * t);

          return state_type{{h, h * v_x}};
        } else if constexpr (dim == 2) {

          const Number &y = point[1];

          const auto elevation =
              eta_ * h_0_ / (a_ * a_) *
              (2. * x * std::cos(omega * t) + 2. * y * std::sin(omega * t));

          h = std::max(elevation - z, Number(0.));
          v_x = -eta_ * omega * std::sin(omega * t);
          v_y = eta_ * omega * std::cos(omega * t);

          return state_type{{h, h * v_x, h * v_y}};

        } else {
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
      const HyperbolicSystemView hyperbolic_system_;

      DEAL_II_ALWAYS_INLINE inline Number
      compute_bathymetry(const dealii::Point<dim> &point) const
      {
        return -h_0_ * (Number(1.) - point.norm_square() / (a_ * a_));
      }

      Number a_;
      Number h_0_;
      Number eta_;
    };

  } // namespace ShallowWaterInitialStates
} // namespace ryujin
