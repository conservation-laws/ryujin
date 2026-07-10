//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

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
      using EquationOfState::pressure;
      using EquationOfState::specific_internal_energy;
      using EquationOfState::speed_of_sound;
      using EquationOfState::temperature;

      PolytropicGas(const std::string &subsection)
          : EquationOfState("polytropic gas", subsection)
      {
        gamma_ = 7. / 5.;
        this->add_parameter("gamma", gamma_, "The ratio of specific heats");

        /*
         * R is the specific gas constant with units [J / (Kg K)]. More details
         * can be found at:
         * https://en.wikipedia.org/wiki/Gas_constant#Specific_gas_constant
         */
        R_ = 287.052874;
        this->add_parameter(
            "gas constant R", R_, "The specific gas constant R");

        cv_ = R_ / (gamma_ - 1.);

        s0_ = 0.;
        this->add_parameter("reference specific entropy",
                            s0_,
                            "The reference specific entropy");

        /* Update the EOS interpolation parameters on parameter read in: */
        ParameterAcceptor::parse_parameters_call_back.connect(
            [this] { cv_ = R_ / (gamma_ - 1.); });
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
       * The temperature is given by
       * \f{align}
       *   T = e / c_v
       * \f}
       */
      double temperature(double /*rho*/, double e) const final
      {
        return e / cv_;
      }

      /**
       * The cold curve bound is given by
       * \f{align}
       *   e_cold = 0
       * \f}
       */
      double cold_curve_bound(double /*rho*/) const final
      {
        return 0.;
      }

      /**
       * The specific entropy is given by
       * \f{align}
       *   s = cv * \ln((gamma - 1) \rho e) -
       *   cv gamma \ln((gamma - 1) cv \rho) + s0
       * \f}
       */
      double specific_entropy(double rho, double e) const final
      {
        const auto gm1 = gamma_ - 1.;
        const auto p = gm1 * rho * e;
        return cv_ * (std::log(p) - gamma_ * std::log(gm1 * cv_ * rho)) + s0_;
      }

      /**
       * The speed of sound is given by
       * \f{align}
       *   c^2 = \gamma * (\gamma - 1) e
       * \f}
       */

      double speed_of_sound(double /*rho*/, double e) const final
      {
        return std::sqrt(gamma_ * (gamma_ - 1.) * e);
      }

    private:
      double gamma_;
      double R_;
      double cv_;
      double s0_;
    };
  } // namespace EquationOfStateLibrary
} // namespace ryujin
