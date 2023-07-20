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
       * The Jones-Wilkins-Lee equation of state
       *
       * @ingroup EulerEquations
       */
      class JonesWilkinsLee : public EquationOfState
      {
      public:
        JonesWilkinsLee(const std::string &subsection)
            : EquationOfState("jones wilkins lee", subsection)
        {
          capA = 0.;
          this->add_parameter("A", capA, "The A constant");

          capB = 0.;
          this->add_parameter("B", capB, "The B constant");

          R1 = 0.;
          this->add_parameter("R1", R1, "The R1 constant");

          R2 = 0.;
          this->add_parameter("R2", R2, "The R2 constant");

          omega = 0.4;
          this->add_parameter("omega", omega, "The Gruneisen coefficient");

          rho0 = 0.;
          this->add_parameter("rho_0", rho0, "The reference density");
        }


        double pressure(const double rho, const double internal_energy) final
        {
          /*
           * p = A(1 - omega / R_1 rho / rho_0) * exp(-R_1 rho_0 / rho) + B(1
           * - omega / R_2 rho/ rho_0) + omega rho * e
           */

          const auto ratio = rho / rho0;

          auto temp = 1. - omega / R1 * ratio;
          const auto first_term = capA * temp * std::exp(-R1 * 1. / ratio);

          temp = 1. - omega / R2 * ratio;
          const auto second_term = capB * temp * std::exp(-R2 * 1. / ratio);

          return first_term + second_term + omega * internal_energy;
        }


        double specific_internal_energy(const double rho, const double p) final
        {
          /*
           * e = (p - [A(1 - omega / R_1 \rho / rho_0) * exp(-R_1 rho_0 / \rho)
           * + B(1 - omega / R_2 \rho/ rho_0)]  / (omega \rho)
           */
          const auto ratio = rho / rho0;

          auto temp = 1. - omega / R1 * ratio;
          const auto first_term = capA * temp * std::exp(-R1 * 1. / ratio);

          temp = 1. - omega / R2 * ratio;
          const auto second_term = capB * temp * std::exp(-R2 * 1. / ratio);

          return (p - (first_term + second_term)) / (rho * omega);
        }

        double material_sound_speed(const double rho, const double p) final
        {
          /*
          c^2 = ...
          */
          auto temp = omega / R1 * rho / rho0;
          temp = omega * (1. - temp) * (1. + 1. / temp) - temp;
          const auto first_term = capA / rho * temp * std::exp(omega / temp);

          temp = omega / R2 * rho / rho0;
          temp = omega * (1. - temp) * (1. + 1. / temp) - temp;
          const auto second_term = capB / rho * temp * std::exp(omega / temp);

          const auto e = specific_internal_energy(rho, p);
          const auto third_term = omega * (omega + 1.) * e;

          return first_term + second_term + third_term;
        }

      private:
        double capA;
        double capB;
        double R1;
        double R2;
        double omega;
        double rho0;
      };
    } // namespace EquationOfStateLibrary
  }   // namespace EulerAEOS
} // namespace ryujin
