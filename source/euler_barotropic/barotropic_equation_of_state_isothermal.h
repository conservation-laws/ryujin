//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "barotropic_equation_of_state.h"

namespace ryujin
{
  namespace BarotropicEquationOfStateLibrary
  {
    /**
     * Isothermal equation of state for the barotropic Euler equations. The
     * specific internal energy, pressure, and speed of sound are given by:
     *
     * \f{align}
     *   e &= c^2 \,\log(\rho), \qquad
     *   p &= c^2 \,\rho, \qquad
     *   a^2 &= c^2.
     * \f}
     *
     * @ingroup EulerEquations
     */
    class Isothermal : public BarotropicEquationOfState
    {
    public:
      Isothermal(const std::string &subsection)
          : BarotropicEquationOfState("isothermal", subsection)
      {
        speed_of_sound_ = 2.;
        add_parameter("speed of sound",
                      speed_of_sound_,
                      "The speed of sound of the isothermal equation of state");
      }

      double specific_internal_energy(double rho) const final
      {
        return speed_of_sound_ * speed_of_sound_ * std::log(rho);
      }

      double pressure(double rho) const final
      {
        return speed_of_sound_ * speed_of_sound_ * rho;
      }

      double speed_of_sound(double /*rho*/) const final
      {
        return speed_of_sound_;
      }

    private:
      double speed_of_sound_;
    };
  } // namespace BarotropicEquationOfStateLibrary
} // namespace ryujin
