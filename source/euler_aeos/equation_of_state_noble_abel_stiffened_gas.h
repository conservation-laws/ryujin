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
        }


        double pressure(const double rho, const double internal_energy) final
        {
          /*
           * p = (\gamma - 1) *  (\rho (e - q))/ (1 - b \rho) - \gamma p_\infty
           */
          const auto num = internal_energy - q_ * rho;
          const auto den = 1. - b_ * rho;
          return (gamma_ - 1.) * num / den - gamma_ * pinf_;
        }


        double specific_internal_energy(const double rho,
                                        const double pressure) final
        {
          /*
           * e = q + (p + \gamma p_\infty) * (1 - b \rho) / (\rho (\gamma -1 ))
           */
          const auto cov = 1. - b_ * rho;
          const auto num = (pressure + gamma_ * pinf_) * cov;
          const auto den = rho * (gamma_ - 1.);
          return num / den + q_;
        }

        double material_sound_speed(const double rho, const double p) final
        {
          /*
           * c^2 = \gamma (p + p_\infty) / (\rho (1 - b \rho))
           */
          const auto cov = 1. - b_ * rho;
          const auto num = gamma_ * (p + pinf_);
          const auto den = rho * cov;
          return std::sqrt(num / den);
        }

      private:
        double gamma_;
        double b_;
        double q_;
        double pinf_;
      };
    } // namespace EquationOfStateLibrary
  }   // namespace EulerAEOS
} /* namespace ryujin */
