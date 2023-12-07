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
     * Smooth vortex problem with and without topography. Without topography,
     * the smooth vortex moves through space. With topography, it is a steady
     * vortex in space. See Section 2.3 in @cite Ricchiuto_Bollermann_2009
     * for details.
     *
     * @ingroup ShallowWaterEquations
     */
    template <typename Description, int dim, typename Number>
    class SmoothVortex : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using HyperbolicSystemView =
          typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      SmoothVortex(const HyperbolicSystem &hyperbolic_system,
                   const std::string subsection)
          : InitialState<Description, dim, Number>("smooth vortex", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        with_bathymetry = false;
        this->add_parameter("with bathymetry",
                            with_bathymetry,
                            "If set to true then the initial profile includes "
                            "the bathymetry for the steady vortex. ");

        depth_ = 1.0;
        this->add_parameter("reference depth", depth_, "Reference water depth");

        mach_number_ = 2.0;
        this->add_parameter(
            "mach number", mach_number_, "Mach number of unsteady vortex");

        beta_ = 0.1;
        this->add_parameter("beta", beta_, "vortex strength beta");
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        const auto gravity = hyperbolic_system_.gravity();

        dealii::Point<2> point_bar;
        point_bar[0] = point[0] - mach_number_ * t;
        point_bar[1] = point[1];

        const Number r_square = Number(point_bar.norm_square());

        /* We assume r0 = 1 */
        const Number factor =
            beta_ / Number(2. * M_PI) *
            exp(Number(0.5) - Number(0.5) * r_square /* /r0^ 2*/);

        Number h =
            depth_ - Number(1. / (2. * gravity /* * r0^2 */)) * factor * factor;

        if (with_bathymetry)
          h -= compute_bathymetry(point);

        const Number u = mach_number_ - factor * Number(point_bar[1]);
        const Number v = factor * Number(point_bar[0]);

        if constexpr (dim == 2)
          return state_type{{h, h * u, h * v}};

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
      const HyperbolicSystemView hyperbolic_system_;

      DEAL_II_ALWAYS_INLINE inline Number
      compute_bathymetry(const dealii::Point<dim> &point) const
      {
        const Number r_square = Number(point.norm_square());

        /* We assume r0 = 1 */
        const Number factor =
            beta_ / Number(2. * M_PI) *
            exp(Number(0.5) - Number(0.5) * r_square /* /r0^ 2*/);

        Number bath = 0.;
        if (with_bathymetry)
          bath = depth_ / 4. * factor;

        return bath;
      }

      bool with_bathymetry;
      Number depth_;
      Number mach_number_;
      Number beta_;
    };

  } // namespace ShallowWaterInitialStates
} // namespace ryujin
