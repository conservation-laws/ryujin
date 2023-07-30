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
     * proposed in Section 5.2 of [Guermond, Nazarov, Popov, Thomas].
     *
     * @todo The set up is as follows:
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

        left_ = 0.1;
        this->add_parameter("left number", left_, "fixme left ");
        right_ = 0.3;
        this->add_parameter("right number", right_, "fixme right ");
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
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

        dealii::Tensor<1, 3, Number> result;
        result[0] = rho;
        result[1] = mach_number_;
        result[2] = pressure_ref_;

        // FIXME: update primitive

        return hyperbolic_system_.from_primitive_state(
            hyperbolic_system_.expand_state(result));
      }

    private:
      const HyperbolicSystemView hyperbolic_system_;

      Number density_ref_;
      Number pressure_ref_;
      Number mach_number_;
      Number left_;
      Number right_;
    };
  } // namespace EulerInitialStates
} // namespace ryujin
