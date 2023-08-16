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
     * A 1D/2D configuration for planar surface flow in a radially-symmetric
     * paraboloid basin without friction. See Section 4.2.2 of
     * @cite swashes_2013 for details.
     *
     * @note The 1D variation is a different than the reference above and has
     * frictions effects.
     *
     * @todo Add variation with friction for 2D case.
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

        if constexpr (dim == 1) {

          length_ = 10000.;
          this->add_parameter(
              "paraboloid length", length_, "Length of 1D paraboloid");

          B_ = 2.;
          this->add_parameter("speed", B_, "The 1D paraboloid speed");

        } else {
          eta_ = 0.5;
          this->add_parameter("eta", eta_, "The eta parameter");
        }
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

        /* Define slightly different profiles for each dimension */

        if constexpr (dim == 1) {

          const Number k = hyperbolic_system_.manning_friction_coefficient();
          const Number p = std::sqrt(8. * g * h_0_) / a_;
          const Number s = std::sqrt(p * p - k * k) / 2.;

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

          h = std::max(htilde, Number(0.));
          v_x = B_ * std::exp(-1. / 2. * k * t) * std::sin(s * t);

          return state_type{{h, h * v_x}};
        } else if constexpr (dim == 2) {

          const Number &y = point[1];

          const Number elevation =
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
        if constexpr (dim == 1)
          return h_0_ / (a_ * a_) * std::pow(point[0] - 0.5 * length_, 2);
        else
          return -h_0_ * (Number(1.) - point.norm_square() / (a_ * a_));
      }

      Number a_;
      Number h_0_;
      Number eta_;
      Number length_;
      Number B_;
    };

  } // namespace ShallowWaterInitialStates
} // namespace ryujin
