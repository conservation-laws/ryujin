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
     * The Noble-Abel-Stiffened gas equation of state
     *
     * @ingroup EulerEquations
     */
    class NobleAbelStiffenedGas : public EquationOfState
    {
    public:
      NobleAbelStiffenedGas(const std::string &subsection)
          : EquationOfState("noble abel stiffened gas", subsection)
      {
        gamma_ = 7. / 5.;
        this->add_parameter("gamma", gamma_, "The ratio of specific heats");

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

        /* Update the interpolation_b_ parameter on parameter read in: */
        ParameterAcceptor::parse_parameters_call_back.connect(
            [this] { this->interpolation_b_ = b_; });
      }

      /**
       * The pressure is given by
       * \f{align}
       *   p = (\gamma - 1) \rho (e - q) / (1 - b \rho) - \gamma p_\infty
       * \f}
       */
      double pressure(double rho, double e) const final
      {
        return (gamma_ - 1.) * rho * (e - q_) / (1. - b_ * rho) -
               gamma_ * pinf_;
      }


      /*
       * The specific internal energy is given by
       * \f{align}
       *   e - q = (p + \gamma p_\infty) * (1 - b \rho) / (\rho (\gamma - 1))
       * \f}
       * \f}
       */
      double specific_internal_energy(double rho, double p) const final
      {
        const auto numerator = (p + gamma_ * pinf_) * (1. - b_ * rho);
        const auto denominator = rho * (gamma_ - 1.);
        return q_ + numerator / denominator;
      }

      /**
       * The speed of sound is given by
       */
      double sound_speed(double rho, double e) const final
      {
        __builtin_trap();
        // FIXME: refactor to new interface
#if 0
          /*
           * \f{align}
           *   c^2 = \gamma (p + p_\infty) / (\rho (1 - b \rho))
           * \f}
           */
          const auto cov = 1. - b_ * rho;
          const auto num = gamma_ * (p + pinf_);
          const auto den = rho * cov;
          return std::sqrt(num / den);
#endif
      }

    private:
      double gamma_;
      double b_;
      double q_;
      double pinf_;
    };
  } // namespace EquationOfStateLibrary
} /* namespace ryujin */
