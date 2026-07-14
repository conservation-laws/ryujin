//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// [LANL Copyright Statement]
// Copyright (C) 2025 by the ryujin authors
// Copyright (C) 2024 - 2025 by Triad National Security, LLC
//

#pragma once

#include "equation_of_state.h"
#include <cmath>

namespace ryujin
{
  namespace EquationOfStateLibrary
  {
    /**
     * The simple MACAW equation of state. See:
     * "A Simple MACAW Equation of State" by Aslam and Lozano (LA-UR-24-32805).
     * This is a "simple" thermodynamically consistent equation of state.
     * The default parameters are taken from Table 1 in reference and were used
     * in calibrating a model for copper.
     *
     * @ingroup EulerEquations
     */
    class SimpleMacaw : public EquationOfState
    {
    public:
      using EquationOfState::pressure;
      using EquationOfState::specific_internal_energy;
      using EquationOfState::speed_of_sound;
      using EquationOfState::temperature;

      SimpleMacaw(const std::string &subsection)
          : EquationOfState("simple macaw", subsection)
      {
        rho0_ = 8.952; // [g/cc]
        this->add_parameter(
            "reference rho0", rho0_, "The reference density at T=0 and P=0");

        T0_ = 150.; // [K]
        this->add_parameter("reference T0", T0_, "The reference temperature");

        Gc_ = 0.5; // [unitless]
        this->add_parameter("Gamma", Gc_, "The Gruneisen parameter");

        capA_ = 7.3; // [Gpa]. This is the "bulk modulus divided by B"
        this->add_parameter("A", capA_, "The A constant");

        // Derivative of the bulk modulus w.r.t pressure at T=0, P=0.
        capB_ = 3.9; // [unitless]
        this->add_parameter("B", capB_, "The B constant");

        // The Dulong-Petit limit of the specific heat at constant volume
        cvInf_ = 3.89e-4; // [kJ / (g K )]
        this->add_parameter("cvInf", cvInf_, "The Dulong-Petit limit of cv");

        /*
         * Update the EOS interpolation parameters on parameter read in and
         * reference volume:
         */
        const auto update_values = [this]() {
          this->interpolation_pinfty_ = capA_ * capB_;
          v0_ = 1. / rho0_;
        };

        this->parse_parameters_call_back.connect(update_values);
        update_values();
      }

      /**
       * Let \f$p_cold = A B ((v / v0)^{-(B+1)} -1)\f$, and let
       * \f$e_cold = A v_0 ((v / v0)^{-B} + B(v / v0) - (B + 1)\f$.
       * The pressure is given by
       * \f{align}
       *   p = p_cold + \Gamma_c * \rho * (e - e_cold)
       * \f}
       */
      double pressure(double rho, double e) const final
      {
        const auto v = 1. / rho;
        const auto ratio = v / v0_;
        const auto p_cold = pressure_cold(v);
        const auto e_cold =
            capA_ * v0_ *
            (std::pow(ratio, -capB_) + ratio * capB_ - (capB_ + 1.));

        return p_cold + Gc_ * rho * (e - e_cold);
      }

      /**
       * Let \f$p_cold = A B ((v / v0)^{-(B+1)} -1)\f$, and let
       * \f$e_cold = A v_0 ((v / v0)^{-B} + B(v / v0) - (B + 1)\f$.
       * The specific internal energy is given by
       * \f{align}
       *   e = (p - p_cold) / (\rho * \Gamma_c) + e_cold
       * \f}
       */
      double specific_internal_energy(double rho, double p) const final
      {
        const auto v = 1. / rho;
        const auto ratio = v / v0_;
        const auto p_cold = pressure_cold(v);
        const auto e_cold =
            capA_ * v0_ *
            (std::pow(ratio, -capB_) + ratio * capB_ - (capB_ + 1.));

        return (p - p_cold) / (Gc_ * rho) + e_cold;
      }

      /**
       * Let \f$e_cold = A v_0 ((v / v0)^{-B} + B(v / v0) - (B + 1)\f$, and let
       * \f$\delta_e = e - e_cold\f$. The temperature is given by
       * \f{align}
       *   c = (\delta_e +
       *        \sqrt(\delta_e(\delta_e + 4 cvInf T_0 (v / v0)^{-Gamma_c})) /
       *        (2 cvInf)
       * \f}
       */
      double temperature(double rho, double e) const final
      {
        const auto v = 1. / rho;
        const auto ratio = v / v0_;
        const auto e_cold =
            capA_ * v0_ *
            (std::pow(ratio, -capB_) + ratio * capB_ - (capB_ + 1.));
        const auto delta_e = e - e_cold;
        const auto radicand =
            delta_e * (delta_e + 4. * cvInf_ * T0_ * std::pow(ratio, -Gc_));
        const auto numerator = delta_e + std::sqrt(radicand);
        const auto denominator = 2. * cvInf_;

        return numerator / denominator;
      }

      /**
       * The cold curve bound is given by
       * \f{align}
       *   e_cold = A v0((v / v0)^{-B} + (v / v0) B - (B+1))
       * \f}
       */
      double cold_curve_bound(double rho) const final
      {
        const auto v = 1. / rho;
        const auto ratio = v / v0_;
        auto e_cold = std::pow(ratio, -capB_) + ratio * capB_ - (capB_ + 1.);
        e_cold *= capA_ * v0_;

        return e_cold;
      }

      /**
       * Let \f$\tau = T * (v / v0)^{\Gamma_c} / T0\f$. The specific entropy is
       * given by
       * \f{align}
       *   s = cvInf ((\tau / (1 + \tau) + \ln(1 + \tau)
       * \f}
       */
      double specific_entropy(double rho, double e) const final
      {
        const auto v = 1. / rho;
        const auto ratio = v / v0_;
        const auto e_cold =
            capA_ * v0_ *
            (std::pow(ratio, -capB_) + ratio * capB_ - (capB_ + 1.));
        const auto delta_e = e - e_cold;
        const auto radicand =
            delta_e * (delta_e + 4. * cvInf_ * T0_ * std::pow(ratio, -Gc_));
        const auto numerator = delta_e + std::sqrt(radicand);
        const auto denominator = 2. * cvInf_;
        const auto T = numerator / denominator;
        const auto tau = T * std::pow(ratio, Gc_) / T0_;

        return cvInf_ * (tau / (1. + tau) + std::log(1. + tau));
      }

      /**
       * Let \f$c_cold = A B ((v / v0)^{-(B+1)} -1)\f$, and let
       * \f$e_cold = A v_0 ((v / v0)^{-B} + B(v / v0) - (B + 1)\f$.
       * The speed of sound is given by
       * \f{align}
       *   c^2 = c_cold + \Gamma_c(\Gamma_c + 1)(e - e_cold)
       * \f}
       */
      double speed_of_sound(double rho, double e) const final
      {
        const auto v = 1. / rho;
        const auto ratio = v / v0_;

        const auto e_cold =
            capA_ * v0_ *
            (std::pow(ratio, -capB_) + ratio * capB_ - (capB_ + 1.));

        // Cold contribution
        const auto c_cold =
            capA_ * capB_ * v0_ * (capB_ + 1.) * std::pow(ratio, -capB_);

        return std::sqrt(c_cold + Gc_ * (Gc_ + 1.) * (e - e_cold));
      }

    private:
      double rho0_;
      double v0_;
      double T0_;
      double capA_;
      double capB_;
      double Gc_;
      double cvInf_;

      double pressure_cold(const double v) const
      {
        const auto ratio = v / v0_;

        auto cold_curve = std::pow(ratio, -capB_ - 1.) - 1.;
        cold_curve *= capA_ * capB_;
        return cold_curve;
      }
    };
  } // namespace EquationOfStateLibrary
} // namespace ryujin
