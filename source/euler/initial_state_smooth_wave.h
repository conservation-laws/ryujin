//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include <initial_state_library.h>
#include <simd.h>

namespace ryujin
{
  namespace EulerInitialStates
  {
    /**
     * This is a generalization of the "Smooth traveling wave" problem first
     * proposed in Section 5.2 of @cite GuermondEtAl2018 for ideal gas EOS.
     * The details for extending to arbitrary equations of state are seen in
     * Section 5.3.1 of @cite ryujin-2023-4.
     *
     * @note This class returns the analytic solution as a function of time
     * @p t and position @p x.
     *
     * @ingroup EulerEquations
     */
    template <typename Description, int dim, typename Number>
    class SmoothWave : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using HyperbolicSystemView =
          typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      using ScalarNumber = typename HyperbolicSystemView::ScalarNumber;

      SmoothWave(const HyperbolicSystem &hyperbolic_system,
                 const std::string subsection)
          : InitialState<Description, dim, Number>("smooth wave", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        density_ref_ = 1.;
        this->add_parameter("reference density",
                            density_ref_,
                            "The material reference density");

        pressure_ref_ = 1.;
        this->add_parameter("reference pressure",
                            pressure_ref_,
                            "The material reference pressure");

        mach_number_ = 1.0;
        this->add_parameter("mach number",
                            mach_number_,
                            "Mach number of traveling smooth wave");

        /* These are the x_0 and x_1 parameters from references above. */
        left_ = 0.1;
        right_ = 0.3;
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        const auto view = hyperbolic_system_.template view<dim, Number>();

        auto point_bar = point;
        point_bar[0] = point_bar[0] - mach_number_ * t;
        const auto x = Number(point_bar[0]);

        const Number polynomial = Number(64) *
                                  ryujin::fixed_power<3>(x - left_) *
                                  ryujin::fixed_power<3>(right_ - x) /
                                  ryujin::fixed_power<6>(right_ - left_);

        /* Define density profile */
        Number rho = density_ref_;
        if (left_ <= point_bar[0] && point_bar[0] <= right_)
          rho = density_ref_ + polynomial;

        state_type initial_state;
        {
          initial_state[0] = rho;
          initial_state[1] = mach_number_;
          initial_state[dim + 1] = pressure_ref_;
        }
        return view.from_initial_state(initial_state);
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;

      Number density_ref_;
      Number pressure_ref_;
      Number mach_number_;
      Number left_;
      Number right_;
    };
  } // namespace EulerInitialStates
} // namespace ryujin
