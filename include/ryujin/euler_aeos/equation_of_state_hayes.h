#pragma once

#include "equation_of_state.h"
#include <cmath>

namespace ryujin
{
  namespace EquationOfStateLibrary
  {
    /**
     * The Hayes equation of state
     *
     * @ingroup EulerEquations
     */
    class Hayes : public EquationOfState
    {
    public:
      using EquationOfState::pressure;
      using EquationOfState::specific_internal_energy;
      using EquationOfState::speed_of_sound;
      using EquationOfState::temperature;

      Hayes(const std::string &subsection)
          : EquationOfState("hayes", subsection)
      {

        rho0_ = 1.844; // [g/cc]
        this->add_parameter("rho_0", rho0_, "The reference density");

        T0_ = 298.15; // [K]
        this->add_parameter("T_0", T0_, "The reference temperature");

        gm0_ = 1.0715848; // [unitless]
        this->add_parameter("gamma_0", gm0_, "The Gruneisen parameter");

        cv_ = 1.11e-3; // [kJ/ (g * K)]
        this->add_parameter(
            "c_v", cv_, "The specific heat capacity at constant volume");

        k0_ = 12.6; // [Gpa]
        this->add_parameter("k_0", k0_, "The bulk modulus");

        // Derivative of bulk modulus w.r.t pressure
        N_ = 5.6; // [unitless]
        this->add_parameter("N", N_, "The exponent N");

        e0_ = 0.; // [kJ / g]
        this->add_parameter(
            "e_0", e0_, "The reference specific internal energy");

        p0_ = 0.; // [Gpa]
        this->add_parameter("p_0", p0_, "The reference pressure");

        s0_ = 0.; // [kJ / (g * K)]
        this->add_parameter("s_0", s0_, "The reference specific entropy");


        const auto update_values = [this]() { v0_ = 1. / rho0_; };

        this->parse_parameters_call_back.connect(update_values);
        update_values();
      }


      /**
       * Let
       * \f{align}
       *   f_c(v) = k0 * v0 / (N * (N - 1)) *
       *            ((v / v0)^{1 - N} - (N - 1)(1 - v / v0) - 1).
       * \f}
       * and let
       * \f{align}
       *   T - T0 = (e - e0 - p0 (v0 - v) - f_c(v)) / cv
       *            + gamma_0 * T0 (1 - v / v0).
       * \f}
       * The pressure is given by
       * \f{align}
       *   p = k0 / N * (v / v0)^{-N} + cv * gamma0 / v0 * (T - T0) - k0 / N +
       *       p0
       * \f}
       */
      double pressure(double rho, double e) const final
      {
        const auto v = 1. / rho;
        const auto ratio = v / v0_;
        const auto f_c = repulsion_of_ions(v);
        const auto delta_T =
            (e - e0_ - p0_ * (v0_ - v) - f_c) / cv_ + gm0_ * T0_ * (1. - ratio);
        const auto first_term = k0_ / N_ * std::pow(ratio, -N_);
        const auto second_term = cv_ * gm0_ / v0_ * delta_T;

        return first_term + second_term - k0_ / N_ + p0_;
      }

      /**
       * Let
       * \f{align}
       *   f_c(v) = k0 * v0 / (N * (N - 1)) *
       *            ((v / v0)^{1 - N} - (N - 1)(1 - v / v0) - 1).
       * \f}
       * The specific internal energy is given by
       * \f{align}
       *   e  = v0 / gamma_0 * (p - p0 - k0 / N (v / v0)^{-N} + k0 / N) +
       *        (p0 v0 - gamma_0 T0 cv)(1 - v / v0) + f_c(v) + e_0
       * \f}
       */
      double specific_internal_energy(double rho, double p) const final
      {
        const auto v = 1. / rho;
        const auto ratio = v / v0_;
        const auto f_c = repulsion_of_ions(v);
        auto scaled_delta_p =
            p - p0_ - k0_ / N_ * std::pow(ratio, -N_) + k0_ / N_;
        scaled_delta_p *= v0_ / gm0_;
        const auto composite_constant = p0_ * v0_ - gm0_ * T0_ * cv_;

        return scaled_delta_p + composite_constant * (1. - ratio) + f_c + e0_;
      }

      /**
       * Let
       * \f{align}
       *   f_c(v) = k0 * v0 / (N * (N - 1)) *
       *            ((v / v0)^{1 - N} - (N - 1)(1 - v / v0) - 1).
       * \f}
       * The temperature is given by
       * \f{align}
       *   T = (e - e0 - p0 (v0 - v) - f_c(v)) / cv + gamma_0 * T0 / v0 (v0 - v)
       *       + T0.
       * \f}
       */
      double temperature(double rho, double e) const final
      {
        const auto v = 1. / rho;
        const auto f_c = repulsion_of_ions(v);
        const auto first_term = (e - e0_ - p0_ * (v0_ - v) - f_c) / cv_;
        const auto second_term = gm0_ * T0_ * (v0_ - v) / v0_;
        return first_term + second_term + T0_;
      }

      /**
       * Let
       * \f{align}
       *   f_c(v) = k0 * v0 / (N * (N - 1)) *
       *            ((v / v0)^{1 - N} - (N - 1)(1 - v / v0) - 1).
       * \f}
       * The cold curve is given by
       * \f{align}
       *   e_cold = p0 (v0 - v) + f_c(v) + e0
       * \f}
       */
      double cold_curve_bound(double rho) const final
      {
        const auto v = 1. / rho;
        const auto f_c = repulsion_of_ions(v);
        return p0_ * (v0_ - v) + f_c + e0_;
      }

      /** Let
       * \f{align}
       *   T = (e - e0 - p0 (v0 - v) - f_c(v)) / cv
       *       + gamma_0 * T0 / v0 (v0 - v) + T_0.
       * \f}
       * The specific entropy is given by
       * \f{align}
       *   s = cv * \ln(T / T0) + cv * gm0 / v0 * (v - v0) + s0
       * \f}
       */
      double specific_entropy(double rho, double e) const final
      {
        const auto v = 1. / rho;
        const auto f_c = repulsion_of_ions(v);
        const auto T = (e - e0_ - p0_ * (v0_ - v) - f_c) / cv_ +
                       gm0_ * T0_ * (v0_ - v) / v0_ + T0_;
        auto s = std::log(T / T0_) + gm0_ * (v - v0_) / v0_;
        s *= cv_;
        return s + s0_;
      }

      /**
       * Let
       * \f{align}
       *   f_c(v) = k0 * v0 / (N * (N - 1)) *
       *            ((v / v0)^{1 - N} - (N - 1)(1 - v / v0) - 1).
       * \f}
       * and let
       * \f{align}
       *   T = (e - e0 - p0 (v0 - v) - f_c(v)) / cv
       *       + gamma_0 * T0 / v0 (v0 - v) + T_0.
       * \f}
       * The speed of sound is given by
       * \f{align}
       *   c^2 = v0 k0 (v / v0)^{1 - N} + cv (gamma0 / v0)^2 v^2 T
       * \f}
       */
      double speed_of_sound(double rho, double e) const final
      {
        const auto v = 1. / rho;
        const auto ratio = v / v0_;
        const auto f_c = repulsion_of_ions(v);
        const auto T = (e - e0_ - p0_ * (v0_ - v) - f_c) / cv_ +
                       gm0_ * T0_ * (v0_ - v) / v0_ + T0_;
        const auto radicand = v0_ * k0_ * std::pow(ratio, 1. - N_) +
                              cv_ * gm0_ * gm0_ * v * v * T / (v0_ * v0_);
        return std::sqrt(radicand);
      }

    private:
      double e0_;
      double k0_;
      double p0_;
      double rho0_;
      double s0_;
      double T0_;
      double gm0_;
      double cv_;
      double N_;
      double v0_;

      /**
       * The repulsion of ions is given by
       * \f{align}
       *   f_c(v) = k0 * v0 / (N * (N - 1)) *
       *            ((v / v0)^{1 - N} - (N - 1)(1 - v / v0) - 1).
       * \f}
       */
      double repulsion_of_ions(const double v) const
      {
        const double ratio = v / v0_;
        double f_c = std::pow(ratio, 1. - N_) - (N_ - 1.) * (1. - ratio) - 1.;
        f_c *= k0_ * v0_ / (N_ * (N_ - 1.));
        return f_c;
      }
    };
  } // namespace EquationOfStateLibrary
} // namespace ryujin
