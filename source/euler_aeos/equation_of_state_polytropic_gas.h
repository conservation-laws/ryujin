//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "equation_of_state.h"

namespace ryujin
{
  namespace EquationOfStateLibrary
  {
    /**
     * The polytropic gas equation of state
     *
     * @ingroup EulerEquations
     */
    class PolytropicGas : public EquationOfState
    {
    public:
      PolytropicGas(const std::string &subsection)
          : EquationOfState("polytropic gas", subsection)
      {
        gamma_ = 7. / 5.;
        this->add_parameter("gamma", gamma_, "The ratio of specific heats");
      }

      /**
       * The pressure is given by
       * \f{align}
       *   p = (\gamma - 1) \rho e
       * \f}
       */
      double pressure(double rho, double e) const final
      {
        return (gamma_ - 1.) * rho * e;
      }

      /**
       * The specific internal energy is given by
       * \f{align}
       *   e = p / (\rho (\gamma - 1))
       * \f}
       */
      double specific_internal_energy(double rho, double p) const final
      {
        return p / (rho * (gamma_ - 1.));
      }

      /**
       * The speed of sound is given by
       * \f{align}
       *   c^2 = \gamma * (\gamma - 1) e
       * \f}
       */
      double sound_speed(double /*rho*/, double e) const final
      {
        return std::sqrt(gamma_ * (gamma_ - 1.) * e);
      }

    private:
      double gamma_;
    };
  } // namespace EquationOfStateLibrary
} // namespace ryujin
