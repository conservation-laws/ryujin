//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "barotropic_equation_of_state.h"

namespace ryujin
{
  namespace BarotropicEquationOfStateLibrary
  {
    /**
     * Isentropic equation of state for the barotropic Euler equations. The
     * specific internal energy, pressure, and speed of sound are given by:
     *
     * \f{align}
     *   e &= \frac{k}{\gamma-1}\,\rho^{\gamma - 1}, \qquad
     *   p &= k\,\rho^{\gamma}, \qquad
     *   a^2 &= \gamma\,k\,\rho^{\gamma - 1}.
     * \f}
     *
     * @ingroup EulerEquations
     */
    class Isentropic : public BarotropicEquationOfState
    {
    public:
      Isentropic(const std::string &subsection)
          : BarotropicEquationOfState("isentropic", subsection)
      {
        k_ = 1.;
        this->add_parameter("k", k_, "Scaling factor k");

        gamma_ = 7. / 5.;
        this->add_parameter("gamma", gamma_, "The ratio of specific heats");
      }

      double specific_internal_energy(double rho) const final
      {
        return k_ / (gamma_ - 1) * std::pow(rho, gamma_ - 1.);
      }

      double pressure(double rho) const final
      {
        return k_ * std::pow(rho, gamma_);
      }

      double speed_of_sound(double rho) const final
      {
        return std::sqrt(gamma_ * k_ * std::pow(rho, gamma_ - 1.));
      }

    private:
      double k_;
      double gamma_;
    };
  } // namespace BarotropicEquationOfStateLibrary
} // namespace ryujin
