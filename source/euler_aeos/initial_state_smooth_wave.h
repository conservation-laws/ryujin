//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"
#include <initial_state.h>

namespace ryujin
{
  namespace EulerAEOS
  {
    /**
     * This is a generalization of the "Smooth traveling wave" problem first
     * proposed in Section 5.2 of [Guermond, Nazarov, Popov, Thomas].
     *
     * @todo The set up is as follows:
     *
     * @ingroup EulerEquations
     */
    template <int dim, typename Number, typename state_type>
    class SmoothWave : public InitialState<dim, Number, state_type>
    {
    public:
      SmoothWave(const HyperbolicSystem &hyperbolic_system,
                 const std::string subsection)
          : InitialState<dim, Number, state_type>("smooth wave", subsection)
          , hyperbolic_system(hyperbolic_system)
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

        const Number polynomial = ryujin::pow(2., 6.) *
                                  ryujin::pow(right_ - left_, -6.) *
                                  ryujin::pow(point_bar[0] - left_, 3.) *
                                  ryujin::pow(right_ - point_bar[0], 3.);

        /* Define density profile */
        Number rho = density_ref_;
        if (left_ <= point_bar[0] && point_bar[0] <= right_)
          rho = density_ref_ + polynomial;

        /* Define specific internal energy from rho and p */
        const Number sie =
            hyperbolic_system.specific_internal_energy_(rho, pressure_ref_);

        dealii::Tensor<1, 3, Number> primitive_temp;
        primitive_temp[0] = rho;
        primitive_temp[1] = mach_number_;
        primitive_temp[2] = sie;

        /* convert to full state */
        const auto full_temp =
            hyperbolic_system.from_primitive_state(primitive_temp);

        return hyperbolic_system.template expand_state<dim>(full_temp);
      }

    private:
      const HyperbolicSystem &hyperbolic_system;

      Number density_ref_;
      Number pressure_ref_;
      Number mach_number_;
      Number left_;
      Number right_;
    };
  } // namespace EulerAEOS
} // namespace ryujin
