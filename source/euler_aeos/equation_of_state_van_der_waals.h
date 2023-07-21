//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "equation_of_state.h"

namespace ryujin
{
  namespace EulerAEOS
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
        double pressure(const double &rho, const double &e) final
        {
          const auto numerator = rho * e + a_ * rho * rho;
          const auto denominator = 1. - b_ * rho;
          return (gamma_ - 1.) * numerator / denominator - a_ * rho * rho;
        }

        /**
         * The specific internal energy is given by
         * \f{align}
         *   \rho e = (p + a \rho^2) * (1 - b \rho) / (\rho (\gamma -1))
         *   - a \rho^2
         * \f}
         */
        double specific_internal_energy(const double &rho,
                                        const double &p) final
        {
          const auto numerator = (p + a_ * rho * rho) * (1. - b_ * rho);
          const auto denominator = rho * (gamma_ - 1.);
          return numerator / denominator - a_ * rho;
        }

        /**
         * The speed of sound is given by
         */
        double sound_speed(const double &rho, const double &e) final
        {
          __builtin_trap();
          // FIXME: refactor to new interface
#if 0
          /*
           * c^2 = \gamma (p + a \rho^2) / (\rho (1 - b \rho)) - 2 a \rho
           */
          const auto cov = 1. - b_ * rho;
          const auto num = gamma_ * (p + a_ * rho * rho);
          const auto den = rho * cov;
          return std::sqrt(num / den - 2. * a_ * rho * rho);
#endif
        }

      private:
        double gamma_;
        double b_;
        double a_;
      };
    } // namespace EquationOfStateLibrary
  }   // namespace EulerAEOS
} /* namespace ryujin */
