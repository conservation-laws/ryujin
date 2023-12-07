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
     * Flow over a bump with a hydraulic jump.
     * See Section 7.2 in @cite GuermondEtAl2018SW for details.
     *
     * @ingroup ShallowWaterEquations
     */
    template <typename Description, int dim, typename Number>
    class FlowOverBump : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using HyperbolicSystemView =
          typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      FlowOverBump(const HyperbolicSystem &hyperbolic_system,
                   const std::string subsection)
          : InitialState<Description, dim, Number>("flow over bump", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        dealii::ParameterAcceptor::parse_parameters_call_back.connect(
            std::bind(&FlowOverBump::parse_parameters_callback, this));

        which_case_ = "transcritical";
        this->add_parameter("flow type",
                            which_case_,
                            "Either 'transcritical' flow with shock "
                            "or 'subsonic' flow.");
      }

      void parse_parameters_callback()
      {
        AssertThrow(which_case_ == "subsonic" || which_case_ == "transcritical",
                    dealii::ExcMessage("Flow type must be 'transcritical' "
                                       "or 'subsonic'. "));
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        const auto g = hyperbolic_system_.gravity();

        const auto x = point[0];

        /* Define constants for transcritical flow */
        const Number xM = 10.;
        const Number xS = 11.7;
        const Number zM = 0.2;
        Number h_inflow = 0.28205279813802181;
        Number q_inflow = 0.18;
        Number cBer =
            zM + 1.5 * ryujin::pow(q_inflow * q_inflow / g, Number(1. / 3.));


        /* Subsonic flow constants */
        if (which_case_ == "subsonic") {
          q_inflow = 4.42;
          h_inflow = 2.;
          cBer = std::pow(q_inflow / h_inflow, 2) / (2. * g) + h_inflow;
        }

        /* General values for Cardano's formula */
        const Number d = q_inflow * q_inflow / (2. * g);
        const Number b = compute_bathymetry(point) - cBer;
        const Number Q = -std::pow(b, 2) / 9.;
        const Number R = -(27. * d + 2. * std::pow(b, 3)) / 54.;
        const Number theta = acos(ryujin::pow(-Q, Number(-1.5)) * R);

        /* Define initial and exact solution */
        const Number h_initial = h_inflow - compute_bathymetry(point);

        /* Explicitly return initial state */
        if (t < 1e-12) {
          return state_type{{h_initial, q_inflow}};
        }

        Number h_exact = 2. * std::sqrt(-Q) * cos(theta / 3.) - b / 3.;
        if (which_case_ == "transcritical") {
          if (xM <= x && x < xS) {
            h_exact = 2. * std::sqrt(-Q) *
                          cos((4. * dealii::numbers::PI + theta) / 3.) -
                      b / 3.;
          } else if (xS < x) {
            h_exact = 0.28205279813802181;
          }
        }

        return state_type{{h_exact, q_inflow}};
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

      DEAL_II_ALWAYS_INLINE
      inline Number compute_bathymetry(const dealii::Point<dim> &point) const
      {
        const auto x = point[0];

        const Number bump = Number(0.2 / 64.) * std::pow(x - Number(8.), 3) *
                            std::pow(Number(12.) - x, 3);

        Number bath = 0.;
        if (8. <= x && x <= 12.)
          bath = bump;
        return bath;
      }

      std::string which_case_;
    };

  } // namespace ShallowWaterInitialStates
} // namespace ryujin
