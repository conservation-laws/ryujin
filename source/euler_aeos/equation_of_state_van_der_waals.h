//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
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
          this->add_parameter("vdw a", b_, "The vdw a constant");

          b_ = 0.;
          this->add_parameter(
              "covolume b", b_, "The maximum compressibility constant");
        }


        double pressure(const double rho, const double internal_energy) final
        {
          /*
           * p = (\gamma - 1) * (\rho * e + a \rho^2)/(1 - b \rho) - a \rho^2
           */
          const auto num = internal_energy + a_ * rho * rho;
          const auto den = 1. - b_ * rho;
          return (gamma_ - 1.) * num / den - a_ * rho * rho;
        }


        double specific_internal_energy(const double rho,
                                        const double pressure) final
        {
          /*
           * rho e = (p + a \rho^2) * (1 - b \rho) / (\rho (\gamma -1))
           * - a \rho^2
           */
          const auto cov = 1. - b_ * rho;
          const auto num = (pressure + a_ * rho * rho) * cov;
          const auto den = rho * (gamma_ - 1.);
          return num / den - a_ * rho;
        }

        double material_sound_speed(const double rho, const double p) final
        {
          /*
           * c^2 = \gamma (p + a \rho^2) / (\rho (1 - b \rho)) - 2 a \rho
           */
          const auto cov = 1. - b_ * rho;
          const auto num = gamma_ * (p + a_ * rho * rho);
          const auto den = rho * cov;
          return std::sqrt(num / den - 2. * a_ * rho * rho);
        }

      private:
        double gamma_;
        double b_;
        double a_;
      };
    } // namespace EquationOfStateLibrary
  }   // namespace EulerAEOS
} /* namespace ryujin */
