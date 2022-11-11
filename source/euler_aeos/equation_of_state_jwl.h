//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

namespace ryujin
{
  namespace EulerAEOS
  {
    namespace EquationOfStateLibrary
    {
      /**
       * The Jones-Wilkins-Lee equation of state
       *
       * @ingroup EquationOfState
       */
      class JonesWilkinsLee : public EquationOfState
      {
      public:
        JonesWilkinsLee(const std::string subsection)
            : EquationOfState("jones-wilkins-lee", subsection)
        {
          capA_ = 0.;
          this->add_parameter("A", capA_, "The A constant");

          capB_ = 0.;
          this->add_parameter("B", capB_, "The B constant");

          R1_ = 0.;
          this->add_parameter("R1", R1_, "The R1 constant");

          R2_ = 0.;
          this->add_parameter("R2", R2_, "The R2 constant");

          omega_ = 0.4;
          this->add_parameter("omega", omega_, "The Gruneisen coefficient");

          rho0_ = 0.;
          this->add_parameter("rho_0", rho0_, "The reference density");
        }

        /* Pressure oracle */
        virtual double
        pressure_oracle(const double rho,
                        const double internal_energy) final override
        {
          /* p = A(1 - omega / R_1 rho / rho_0) * exp(-R_1 rho_0 / rho) + B(1
           * - omega / R_2 rho/ rho_0) + omega rho * e */

          const double ratio = rho / rho0_;

          double temp = 1. - omega_ / R1_ * ratio;
          const double first_term = capA_ * temp * std::exp(-R1_ * 1. / ratio);

          temp = 1. - omega_ / R2_ * ratio;
          const double second_term = capB_ * temp * std::exp(-R2_ * 1. / ratio);

          return first_term + second_term + omega_ * internal_energy;
        }

        /* Sie from rho and p */
        virtual double sie_from_rho_p(const double rho,
                                      const double pressure) final override
        {
          const double ratio = rho / rho0_;

          double temp = 1. - omega_ / R1_ * ratio;
          const double first_term = capA_ * temp * std::exp(-R1_ * 1. / ratio);

          temp = 1. - omega_ / R2_ * ratio;
          const double second_term = capB_ * temp * std::exp(-R2_ * 1. / ratio);


          return (pressure - (first_term + second_term)) / (rho * omega_);
        }

      private:
        double capA_;
        double capB_;
        double R1_;
        double R2_;
        double omega_;
        double rho0_;
      };
    } // namespace EquationOfStateLibrary
  } // namespace EulerAEOS
} // namespace ryujin
