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
     * The Jones-Wilkins-Lee equation of state. See (16a) in:
     * "JWL Equation of State" by Ralph Menikoff (LA-UR-15-29536). We are
     * assuming that the reference temperature T_r is chosen such that E_r = 0.
     * Default parameters taken from Table 1 in reference.
     *
     * @ingroup EulerEquations
     */
    class JonesWilkinsLee : public EquationOfState
    {
    public:
      JonesWilkinsLee(const std::string &subsection)
          : EquationOfState("jones wilkins lee", subsection)
      {
        capA = 6.3207e13; // [Pa]
        this->add_parameter("A", capA, "The A constant");

        capB = -4.472e9; // [Pa]
        this->add_parameter("B", capB, "The B constant");

        R1 = 11.3; // [unitless]
        this->add_parameter("R1", R1, "The R1 constant");

        R2 = 1.13; // [unitless]
        this->add_parameter("R2", R2, "The R2 constant");

        omega = 0.8938; // [unitless]
        this->add_parameter("omega", omega, "The Gruneisen coefficient");

        rho_0 = 1895; // [Kg / m^3]
        this->add_parameter("rho_0", rho_0, "The reference density");

        q_0 = 0.0; // [J / Kg]
        this->add_parameter("q_0", q_0, "The specific internal energy offset");

        cv_ = 2487. / rho_0; // [J / (Kg * K)]
        this->add_parameter(
            "c_v", cv_, "The specific heat capacity at constant volume");
      }

      /**
       * The pressure is given by
       * \f{align}
       *   p = A(1 - \omega / R_1 \rho / \rho_0) e^{(-R_1 \rho_0 / \rho)}
       *     + B(1 - \omega / R_2 \rho/ \rho_0) e^{(-R_2 \rho_0 / \rho)}
       *     + \omega \rho (e + q_0)
       * \f}
       */
      double pressure(double rho, double e) const final
      {

        const auto ratio = rho / rho_0;

        const auto first_term =
            capA * (1. - omega / R1 * ratio) * std::exp(-R1 * 1. / ratio);
        const auto second_term =
            capB * (1. - omega / R2 * ratio) * std::exp(-R2 * 1. / ratio);

        return first_term + second_term + omega * rho * (e + q_0);
      }

      /**
       * The specific internal energy is given by
       * \f{align}
       *   \omega \rho e = p
       *   - A(1 - \omega / R_1 \rho / \rho_0) e^{(-R_1 \rho_0 / \rho)}
       *   - B(1 - \omega / R_2 \rho/ \rho_0) e^{(-R_2 \rho_0 / \rho)}
       * \f}
       */
      double specific_internal_energy(double rho, double p) const final
      {
        const auto ratio = rho / rho_0;

        const auto first_term =
            capA * (1. - omega / R1 * ratio) * std::exp(-R1 * 1. / ratio);
        const auto second_term =
            capB * (1. - omega / R2 * ratio) * std::exp(-R2 * 1. / ratio);

        return (p - first_term - second_term) / (rho * omega);
      }

      /**
       * The temperature is given by
       * \f{align}
       *   c_v T = e + q_0 - 1 / \rho_0 * (A / R_1 * e^{(-R_1 \rho_0 / \rho)}
       *         + B / R_2 * e^{(-R_2 \rho_0 / \rho)})
       * \f}
       */
      double temperature(double rho, double e) const final
      {
        /* Using (16a) of LA-UR-15-29536 */
        const auto ratio = rho / rho_0;

        const auto first_term = capA / R1 * std::exp(-R1 * 1. / ratio);
        const auto second_term = capB / R2 * std::exp(-R2 * 1. / ratio);

        return (e + q_0 - 1. / rho_0 * (first_term + second_term)) / cv_;
      }

      /**
       * The speed of sound is given by
       */
      double speed_of_sound(double rho, double e) const final
      {
        /* FIXME: Need to cross reference with literature */

        const auto t1 = omega * rho / (R1 * rho_0);
        const auto factor1 = omega * (1. - t1) * (1. + 1. / t1) - t1;
        const auto first_term =
            capA / rho * factor1 * std::exp(-1. / t1 / omega);

        const auto t2 = omega * rho / (R2 * rho_0);
        const auto factor2 = omega * (1. - t2) * (1. + 1. / t2) - t2;
        const auto second_term =
            capB / rho * factor2 * std::exp(-1. / t2 / omega);

        const auto third_term = omega * (omega + 1.) * e;

        return std::sqrt(first_term + second_term + third_term);
      }

    private:
      double capA;
      double capB;
      double R1;
      double R2;
      double omega;
      double rho_0;
      double q_0;
      double cv_;
    };
  } // namespace EquationOfStateLibrary
} // namespace ryujin
