//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 by the ryujin authors
//

#pragma once

#include "equation_of_state.h"

namespace ryujin
{
  namespace EquationOfStateLibrary
  {
    /**
     * The Noble-Abel-Stiffened gas equation of state
     *
     * @ingroup EulerEquations
     */
    class NobleAbelStiffenedGas : public EquationOfState
    {
    public:
      using EquationOfState::pressure;
      using EquationOfState::specific_internal_energy;
      using EquationOfState::speed_of_sound;
      using EquationOfState::temperature;

      NobleAbelStiffenedGas(const std::string &subsection)
          : EquationOfState("noble abel stiffened gas", subsection)
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

        b_ = 0.;
        this->add_parameter(
            "covolume b", b_, "The maximum compressibility constant");

        q_ = 0.;
        this->add_parameter("reference specific internal energy",
                            q_,
                            "The reference specific internal energy");

        pinf_ = 0.;
        this->add_parameter(
            "reference pressure", pinf_, "The reference pressure p infinity");

        /* Update the EOS interpolation parameters on parameter read in: */
        ParameterAcceptor::parse_parameters_call_back.connect([this] {
          this->interpolation_b_ = b_;
          this->interpolation_pinfty_ = pinf_;
        });
      }

      /**
       * The pressure is given by
       * \f{align}
       *   p = (\gamma - 1) \rho (e - q) / (1 - b \rho) - \gamma p_\infty
       * \f}
       */
      double pressure(double rho, double e) const final
      {
        return (gamma_ - 1.) * rho * (e - q_) / (1. - b_ * rho) -
               gamma_ * pinf_;
      }


      /**
       * The specific internal energy is given by
       * \f{align}
       *   e - q = (p + \gamma p_\infty) * (1 - b \rho) / (\rho (\gamma - 1))
       * \f}
       */
      double specific_internal_energy(double rho, double p) const final
      {
        const auto numerator = (p + gamma_ * pinf_) * (1. - b_ * rho);
        const auto denominator = rho * (gamma_ - 1.);
        return q_ + numerator / denominator;
      }

      /**
       * The temperature is given by
       * \f{align}
       *   T = (e - q - p_\infty (1 / rho - b)) / c_v
       * \f}
       */
      double temperature(double rho, double e) const final
      {
        return (e - q_ - pinf_ * (1. / rho - b_)) / cv_;
      }

      /**
       * Let \f$X = (1 - b \rho)\f$. The speed of sound is given by
       * \f{align}
       *   c^2 = \frac{\gamma (p + p_\infty)}{\rho X}
       *       = \frac{\gamma (\gamma -1)[\rho (e - q) - p_\infty X]}{\rho X^2}
       * \f}
       */
      double speed_of_sound(double rho, double e) const final
      {
        const auto covolume = 1. - b_ * rho;
        auto numerator = (rho * (e - q_) - pinf_ * covolume) / rho;
        numerator *= gamma_ * (gamma_ - 1.);
        return std::sqrt(numerator) / covolume;
      }

    private:
      double gamma_;
      double R_;
      double cv_;
      double b_;
      double q_;
      double pinf_;
    };
  } // namespace EquationOfStateLibrary
} /* namespace ryujin */
