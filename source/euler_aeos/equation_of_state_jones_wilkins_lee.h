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
     * The Jones-Wilkins-Lee equation of state. See "JWL Equation of State" by
     * Ralph Menikoff (LA-UR-15-29536)
     *
     * @ingroup EulerEquations
     */
    class JonesWilkinsLee : public EquationOfState
    {
    public:
      JonesWilkinsLee(const std::string &subsection)
          : EquationOfState("jones wilkins lee", subsection)
      {
        capA = 0.0;
        this->add_parameter("A", capA, "The A constant");

        capB = 0.0;
        this->add_parameter("B", capB, "The B constant");

        R1 = 1.0;
        this->add_parameter("R1", R1, "The R1 constant");

        R2 = 1.0;
        this->add_parameter("R2", R2, "The R2 constant");

        omega = 0.4;
        this->add_parameter("omega", omega, "The Gruneisen coefficient");

        rho_0 = 1.0;
        this->add_parameter("rho_0", rho_0, "The reference density");

        q_0 = 0.0;
        this->add_parameter("q_0", q_0, "The specific internal energy offset");

        cv_ = 718.;
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
        /* Using (16a) of LA-UR-15-29536 (need to verify if valid)*/
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
        // FIXME: This needs to be verified...

        const auto t1 = omega * rho / (R1 * rho_0);
        const auto factor1 = omega * (1. - t1) * (1. + 1. / t1) - t1;
        const auto first_term =
            capA / rho * factor1 * std::exp(omega / factor1);

        const auto t2 = omega * rho / (R2 * rho_0);
        const auto factor2 = omega * (1. - t2) * (1. + 1. / t2) - t2;
        const auto second_term =
            capB / rho * factor2 * std::exp(omega / factor2);

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
