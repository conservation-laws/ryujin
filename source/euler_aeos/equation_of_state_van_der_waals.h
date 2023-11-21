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
     * The Van der Waals equation of state
     *
     * @ingroup EulerEquations
     */
    class VanDerWaals : public EquationOfState
    {
    public:
      VanDerWaals(const std::string &subsection)
          : EquationOfState("van der waals", subsection)
      {
        gamma_ = 7. / 5.;
        this->add_parameter("gamma", gamma_, "The ratio of specific heats");

        a_ = 0.;
        this->add_parameter("vdw a", a_, "The vdw a constant");

        b_ = 0.;
        this->add_parameter(
            "covolume b", b_, "The maximum compressibility constant");

        R_ = 0.4;
        this->add_parameter(
            "gas constant R", R_, "The specific gas constant R");

        /* Update the interpolation_b_ parameter on parameter read in: */
        ParameterAcceptor::parse_parameters_call_back.connect(
            [this] { this->interpolation_b_ = b_; });
      }

      /**
       * The pressure is given by
       * \f{align}
       *   p = (\gamma - 1) * (\rho * e + a \rho^2)/(1 - b \rho) - a \rho^2
       * \f}
       */
      double pressure(double rho, double e) const final
      {
        const auto intermolecular = a_ * rho * rho;
        const auto numerator = rho * e + intermolecular;
        const auto covolume = 1. - b_ * rho;
        return (gamma_ - 1.) * numerator / covolume - intermolecular;
      }

      /**
       * The specific internal energy is given by
       * \f{align}
       *   \rho e = (p + a \rho^2) * (1 - b \rho) / (\rho (\gamma -1))
       *   - a \rho^2
       * \f}
       */
      double specific_internal_energy(double rho, double p) const final
      {
        const auto intermolecular = a_ * rho * rho;
        const auto covolume = 1. - b_ * rho;
        const auto numerator = (p + intermolecular) * covolume;
        const auto denominator = rho * (gamma_ - 1.);
        return numerator / denominator - a_ * rho;
      }

      /**
       * The temperature is given by
       * \f{align}
       *   T = (\gamma - 1) / R (e + a \rho)
       * \f}
       */
      double temperature(double rho, double e) const final
      {
        return (gamma_ - 1.) / R_ * (e + a_ * rho);
      }

      /**
       * The speed of sound is given by
       * \f{align}
       *   c^2 = \frac{\gamma (\gamma -1) (e + a \rho)}{(1 - b\rho)^2}
       *   - 2a\rho.
       * \f}
       */
      double speed_of_sound(double rho, double e) const final
      {
        const auto covolume = 1. - b_ * rho;
        const auto numerator = gamma_ * (gamma_ - 1.) * (e + a_ * rho);
        return std::sqrt(numerator / (covolume * covolume) - 2. * a_ * rho);
      }

    private:
      double gamma_;
      double b_;
      double a_;
      double R_;
    };
  } // namespace EquationOfStateLibrary
} /* namespace ryujin */
