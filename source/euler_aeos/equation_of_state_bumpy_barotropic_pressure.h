//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "equation_of_state.h"

namespace ryujin
{
  namespace EquationOfStateLibrary
  {
    /**
     * A composite equation of state of the form
     * \f{align}
     *   p\;=\;p_{\text{h}}(\rho,e_{\text{h}})\,+\,p_{\text{b}}(\rho),
     *   \qquad
     *   e\;=\;e_{\text{h}}\,+\,e_{\text{b}}(\rho),
     *   \qquad
     *   p_{\text{b}}(\rho)\;=\;\rho^2\,\partial_\rho e_{\text{b}}(\rho),
     * \f}
     * in which the hydrodynamical constituent
     * \f$p_{\text{h}},\,e_{\text{h}}\f$ is given by a Noble-Abel stiffened
     * gas and the barotropic constituent
     * \f$p_{\text{b}},\,e_{\text{b}}\f$ is a synthetic "bump" that
     * concentrates the barotropic speed of sound in a narrow interval of
     * width \f$\varepsilon\f$ centered at a reference density \f$\rho_0\f$:
     * \f{align}
     *   a_{\text{b}}^2(\rho)\;=\;
     *   \frac{c_0^2}{\big((\rho-\rho_0)^2+\varepsilon\big)^{3/2}}.
     * \f}
     * The bump renders the total pressure \f$p\f$ nonconvex so that the
     * Riemann problem can develop composite shock-rarefaction waves, while
     * hyperbolicity is maintained.
     *
     * The barotropic constituent carries no entropy and no temperature:
     * \f$T\,\mathrm{d}s = \mathrm{d}e + p\,\mathrm{d}v =
     * \mathrm{d}e_{\text{h}} + p_{\text{h}}\,\mathrm{d}v\f$, so that
     * \f$s=s_{\text{h}}\f$ and \f$T=T_{\text{h}}\f$. Consequently the
     * isentropes of the composite equation of state are the hydrodynamical
     * isentropes and the total speed of sound is given by
     * \f$a^2=a_{\text{h}}^2+a_{\text{b}}^2\f$.
     *
     * @note All functions of this class take and return the *total*
     * specific internal energy @p e and the *total* pressure @p p. The
     * hydrodynamical constituents are recovered as
     * \f$e_{\text{h}}=e-e_{\text{b}}(\rho)\f$ and
     * \f$p_{\text{h}}=p-p_{\text{b}}(\rho)\f$.
     *
     * @ingroup EulerEquations
     */
    class BumpyBarotropicPressure : public EquationOfState
    {
    public:
      using EquationOfState::pressure;
      using EquationOfState::specific_internal_energy;
      using EquationOfState::speed_of_sound;
      using EquationOfState::temperature;

      BumpyBarotropicPressure(const std::string &subsection)
          : EquationOfState("bumpy barotropic pressure", subsection)
      {
        /*
         * Parameters of the hydrodynamical (Noble-Abel stiffened gas)
         * constituent:
         */

        gamma_ = 7. / 5.;
        this->add_parameter("gamma", gamma_, "The ratio of specific heats");

        /*
         * R is the specific gas constant with units [J / (Kg K)]. More details
         * can be found at:
         * https://en.wikipedia.org/wiki/Gas_constant#Specific_gas_constant
         */
        R_ = 287.052874;
        this->add_parameter(
            "gas constant R", R_, "The specific gas constant R");

        cv_ = R_ / (gamma_ - 1.);

        b_ = 0.;
        this->add_parameter(
            "covolume b", b_, "The maximum compressibility constant");

        q_ = 0.;
        this->add_parameter("reference specific internal energy",
                            q_,
                            "The reference specific internal energy");

        pinf_ = 0.;
        this->add_parameter(
            "reference pressure", pinf_, "The reference pressure p infinity");

        s0_ = 0.;
        this->add_parameter("reference specific entropy",
                            s0_,
                            "The reference specific entropy");

        /*
         * Parameters of the barotropic constituent:
         */

        rho_0_ = 1.;
        this->add_parameter("barotropic reference density",
                            rho_0_,
                            "The density rho_0 at which the barotropic speed "
                            "of sound is centered");

        c_0_ = 1.;
        this->add_parameter("barotropic sound speed",
                            c_0_,
                            "The strength c_0 of the barotropic speed of "
                            "sound");

        eps_ = 1.e-2;
        this->add_parameter("barotropic bump width",
                            eps_,
                            "The width epsilon of the barotropic bump");

        e_0_ = 0.;
        this->add_parameter("barotropic reference specific internal energy",
                            e_0_,
                            "The (arbitrary) additive constant e_0 of the "
                            "barotropic specific internal energy");

        /*
         * Update the EOS interpolation parameters on parameter read in
         * and specific heat at constant volume:
         *
         * @note The interpolation parameters are the ones of the
         * hydrodynamical constituent. They drive the surrogate NASG
         * interpolation performed in HyperbolicSystemView
         * (surrogate_gamma(), surrogate_pressure(),
         * surrogate_specific_entropy(), ...) which is unaware of the
         * barotropic contribution.
         */
        ParameterAcceptor::parse_parameters_call_back.connect([this] {
          this->covolume_constant_ = b_;
          this->interpolation_pinfty_ = pinf_;
          this->interpolation_q_ = q_;
          cv_ = R_ / (gamma_ - 1.);
        });
      }

      /**
       * The total pressure is given by
       * \f{align}
       *   p = p_{\text{h}}(\rho, e - e_{\text{b}}(\rho)) + p_{\text{b}}(\rho)
       * \f}
       */
      double pressure(double rho, double e) const final
      {
        const auto e_h = e - barotropic_specific_internal_energy(rho);
        return hydrodynamic_pressure(rho, e_h) + barotropic_pressure(rho);
      }


      /**
       * The total specific internal energy is given by
       * \f{align}
       *   e = e_{\text{h}}(\rho, p - p_{\text{b}}(\rho)) + e_{\text{b}}(\rho)
       * \f}
       */
      double specific_internal_energy(double rho, double p) const final
      {
        const auto p_h = p - barotropic_pressure(rho);
        return hydrodynamic_specific_internal_energy(rho, p_h) +
               barotropic_specific_internal_energy(rho);
      }

      /**
       * The barotropic constituent is athermal, so that the temperature is
       * the hydrodynamical one,
       * \f{align}
       *   T = T_{\text{h}}(\rho, e - e_{\text{b}}(\rho))
       * \f}
       */
      double temperature(double rho, double e) const final
      {
        const auto e_h = e - barotropic_specific_internal_energy(rho);
        return hydrodynamic_temperature(rho, e_h);
      }

      /**
       * The admissible set is characterized by \f$e_{\text{h}}\ge
       * e_{\text{h},\text{cold}}(\rho)\f$ and thus
       * \f{align}
       *   e_{\text{cold}} =
       *   e_{\text{h},\text{cold}}(\rho) + e_{\text{b}}(\rho)
       * \f}
       */
      double cold_curve_bound(double rho) const final
      {
        return hydrodynamic_cold_curve_bound(rho) +
               barotropic_specific_internal_energy(rho);
      }

      /**
       * The barotropic constituent is isentropic, so that the specific
       * entropy is the hydrodynamical one,
       * \f{align}
       *   s = s_{\text{h}}(\rho, e - e_{\text{b}}(\rho))
       * \f}
       */
      double specific_entropy(double rho, double e) const final
      {
        const auto e_h = e - barotropic_specific_internal_energy(rho);
        return hydrodynamic_specific_entropy(rho, e_h);
      }

      /**
       * The isentropes of the composite equation of state coincide with the
       * hydrodynamical isentropes, so that the speed of sound is given by
       * \f{align}
       *   a^2 = a_{\text{h}}^2(\rho, e - e_{\text{b}}(\rho))
       *       + a_{\text{b}}^2(\rho)
       * \f}
       */
      double speed_of_sound(double rho, double e) const final
      {
        const auto e_h = e - barotropic_specific_internal_energy(rho);
        return std::sqrt(hydrodynamic_sound_speed_squared(rho, e_h) +
                         barotropic_sound_speed_squared(rho));
      }

    private:
      /**
       * The barotropic speed of sound squared,
       * \f{align}
       *   a_{\text{b}}^2 =
       *   \frac{c_0^2}{\big((\rho-\rho_0)^2+\varepsilon\big)^{3/2}}
       * \f}
       */
      double barotropic_sound_speed_squared(double rho) const
      {
        const auto radicand = (rho - rho_0_) * (rho - rho_0_) + eps_;
        return c_0_ * c_0_ / (radicand * std::sqrt(radicand));
      }

      /**
       * Integrating \f$a_{\text{b}}^2=\partial_\rho p_{\text{b}}\f$ subject
       * to \f$p_{\text{b}}(0)=0\f$ yields
       * \f{align}
       *   p_{\text{b}} = \frac{c_0^2}{\varepsilon}\bigg(
       *     \frac{\rho-\rho_0}{\sqrt{(\rho-\rho_0)^2+\varepsilon}}
       *     + \frac{\rho_0}{\sqrt{\rho_0^2+\varepsilon}}\bigg)
       * \f}
       */
      double barotropic_pressure(double rho) const
      {
        const auto s = std::sqrt((rho - rho_0_) * (rho - rho_0_) + eps_);
        const auto s_0 = std::sqrt(rho_0_ * rho_0_ + eps_);
        return c_0_ * c_0_ / eps_ * ((rho - rho_0_) / s + rho_0_ / s_0);
      }

      /**
       * Integrating \f$p_{\text{b}}=\rho^2\,\partial_\rho e_{\text{b}}\f$
       * yields, with \f$s=\sqrt{(\rho-\rho_0)^2+\varepsilon}\f$ and
       * \f$s_0=\sqrt{\rho_0^2+\varepsilon}\f$,
       * \f{align}
       *   e_{\text{b}} =
       *   \frac{c_0^2\,\rho_0}{\varepsilon\,s_0^2\,\rho}\,\big(s-s_0\big)
       *   + \frac{2\,c_0^2}{s_0^3}\,
       *     \operatorname{artanh}\Big(\frac{\rho-s}{s_0}\Big)
       *   + e_0
       * \f}
       *
       * @note We evaluate the argument of the area hyperbolic tangent as
       * \f$(2\rho\rho_0-\rho_0^2-\varepsilon)/(\rho+s)/s_0\f$, which is
       * algebraically equivalent but avoids the cancellation of
       * \f$\rho-s\f$ for \f$\rho\gg\rho_0\f$.
       */
      double barotropic_specific_internal_energy(double rho) const
      {
        const auto s = std::sqrt((rho - rho_0_) * (rho - rho_0_) + eps_);
        const auto s_0_squared = rho_0_ * rho_0_ + eps_;
        const auto s_0 = std::sqrt(s_0_squared);

        const auto first_term =
            c_0_ * c_0_ * rho_0_ * (s - s_0) / (eps_ * s_0_squared * rho);

        const auto argument =
            (2. * rho * rho_0_ - s_0_squared) / ((rho + s) * s_0);
        const auto second_term =
            2. * c_0_ * c_0_ / (s_0_squared * s_0) * std::atanh(argument);

        return first_term + second_term + e_0_;
      }

      /**
       * The hydrodynamical (Noble-Abel stiffened gas) pressure is given by
       * \f{align}
       *   p_{\text{h}} =
       *   (\gamma - 1) \rho (e_{\text{h}} - q) / (1 - b \rho)
       *   - \gamma p_\infty
       * \f}
       */
      double hydrodynamic_pressure(double rho, double e_h) const
      {
        return (gamma_ - 1.) * rho * (e_h - q_) / (1. - b_ * rho) -
               gamma_ * pinf_;
      }

      /**
       * The hydrodynamical specific internal energy is given by
       * \f{align}
       *   e_{\text{h}} - q =
       *   (p_{\text{h}} + \gamma p_\infty) (1 - b \rho) / (\rho (\gamma - 1))
       * \f}
       */
      double hydrodynamic_specific_internal_energy(double rho, double p_h) const
      {
        const auto numerator = (p_h + gamma_ * pinf_) * (1. - b_ * rho);
        const auto denominator = rho * (gamma_ - 1.);
        return q_ + numerator / denominator;
      }

      /**
       * The hydrodynamical temperature is given by
       * \f{align}
       *   T = (e_{\text{h}} - q - p_\infty (1 / \rho - b)) / c_v
       * \f}
       */
      double hydrodynamic_temperature(double rho, double e_h) const
      {
        return (e_h - q_ - pinf_ * (1. / rho - b_)) / cv_;
      }

      /**
       * The hydrodynamical cold curve bound is given by
       * \f{align}
       *   e_{\text{h},\text{cold}} = q + p_\infty (1 / \rho - b)
       * \f}
       */
      double hydrodynamic_cold_curve_bound(double rho) const
      {
        return q_ + pinf_ * (1. / rho - b_);
      }

      /**
       * The hydrodynamical specific entropy is given by
       * \f{align}
       *   p_{\text{h}} + p_\infty =
       *   (\gamma - 1)((e_{\text{h}} - q) - p_\infty (1 / \rho - b))
       *   / (1 / \rho - b)
       * \f}
       * \f{align}
       *   s = c_v \ln(p_{\text{h}} + p_\infty) -
       *   c_v \gamma \ln(\frac{(\gamma - 1) c_v}{1 / \rho - b}) + s_0
       * \f}
       */
      double hydrodynamic_specific_entropy(double rho, double e_h) const
      {
        const auto covolume_term = 1. / rho - b_;
        const auto p_plus_pinf = (gamma_ - 1.) *
                                 ((e_h - q_) - pinf_ * covolume_term) /
                                 covolume_term;
        const auto first_term = cv_ * std::log(p_plus_pinf);
        const auto second_term =
            cv_ * gamma_ * std::log((gamma_ - 1.) * cv_ / covolume_term);
        return first_term - second_term + s0_;
      }

      /**
       * Let \f$X = (1 - b \rho)\f$. The hydrodynamical speed of sound
       * squared is given by
       * \f{align}
       *   a_{\text{h}}^2 = \frac{\gamma (p_{\text{h}} + p_\infty)}{\rho X}
       *       = \frac{\gamma (\gamma -1)
       *         [\rho (e_{\text{h}} - q) - p_\infty X]}{\rho X^2}
       * \f}
       */
      double hydrodynamic_sound_speed_squared(double rho, double e_h) const
      {
        const auto covolume = 1. - b_ * rho;
        auto result =
            (rho * (e_h - q_) - pinf_ * covolume) / (covolume * covolume * rho);
        result *= gamma_ * (gamma_ - 1.);
        return result;
      }

      double gamma_;
      double R_;
      double cv_;
      double b_;
      double q_;
      double pinf_;
      double s0_;

      double rho_0_;
      double c_0_;
      double eps_;
      double e_0_;
    };
  } // namespace EquationOfStateLibrary
} /* namespace ryujin */
