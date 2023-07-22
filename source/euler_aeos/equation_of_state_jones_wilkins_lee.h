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

        rho0 = 1.0;
        this->add_parameter("rho_0", rho0, "The reference density");
      }

      /**
       * The pressure is given by
       * \f{align}
       *   p = A(1 - \omega / R_1 \rho / \rho_0) e^{(-R_1 \rho_0 / \rho)}
       *     + B(1 - \omega / R_2 \rho/ \rho_0) e^{(-R_2 \rho_0 / \rho)}
       *     + \omega \rho e
       * \f}
       */
      double pressure(double rho, double e) const final
      {

        const auto ratio = rho / rho0;

        const auto first_term =
            capA * (1. - omega / R1 * ratio) * std::exp(-R1 * 1. / ratio);
        const auto second_term =
            capB * (1. - omega / R2 * ratio) * std::exp(-R2 * 1. / ratio);

        return first_term + second_term + omega * rho * e;
      }

      /*
       * The specific internal energy is given by
       * \f{align}
       *   \omega \rho e = p
       *   - A(1 - \omega / R_1 \rho / \rho_0) e^{(-R_1 \rho_0 / \rho)}
       *   - B(1 - \omega / R_2 \rho/ \rho_0) e^{(-R_2 \rho_0 / \rho)}
       * \f}
       */
      double specific_internal_energy(double rho, double p) const final
      {
        const auto ratio = rho / rho0;

        const auto first_term =
            capA * (1. - omega / R1 * ratio) * std::exp(-R1 * 1. / ratio);
        const auto second_term =
            capB * (1. - omega / R2 * ratio) * std::exp(-R2 * 1. / ratio);

        return (p - first_term - second_term) / (rho * omega);
      }

      /**
       * The speed of sound is given by
       */
      double speed_of_sound(double rho, double e) const final
      {
        // FIXME: This needs to be verified...

        const auto t1 = omega * rho / (R1 * rho0);
        const auto factor1 = omega * (1. - t1) * (1. + 1. / t1) - t1;
        const auto first_term =
            capA / rho * factor1 * std::exp(omega / factor1);

        const auto t2 = omega * rho / (R2 * rho0);
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
      double rho0;
    };
  } // namespace EquationOfStateLibrary
} // namespace ryujin
