//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

namespace ryujin
{
  namespace EquationOfStateLibrary
  {
    /**
     * The polytropic gas equation of state
     *
     * @ingroup EquationOfState
     */
    class PolytropicGas : public EquationOfState
    {
    public:
      PolytropicGas(const std::string subsection)
          : EquationOfState("polytropic gas", subsection)
      {
        gamma_ = 7. / 5.;
        this->add_parameter("gamma", gamma_, "The ratio of specific heats");
      }

      virtual double
      pressure_oracle(const double /*rho*/,
                      const double internal_energy) final override
      {
        return (gamma_ - 1.) * internal_energy;
      }

    private:
      double gamma_;
    };

    /**
     * The noble-abel-stiffened gas equation of state
     *
     * @ingroup EquationOfState
     */
    class NobleAbleStiffenedGas : public EquationOfState
    {
    public:
      NobleAbleStiffenedGas(const std::string subsection)
          : EquationOfState("noble-able-stiffened gas", subsection)
      {
        gamma_ = 7. / 5.;
        this->add_parameter("gamma", gamma_, "The ratio of specific heats");

        b_ = 0.;
        this->add_parameter(
            "covolume b", b_, "The maximum compressibility constant");

        q_ = 0.;
        this->add_parameter(
            "reference sie q", q_, "The reference specific internal energy");

        pinf_ = 0.;
        this->add_parameter(
            "reference pressure", pinf_, "The reference pressure p infinity");
      }

      virtual double
      pressure_oracle(const double /*rho*/,
                      const double internal_energy) final override
      {
        return (gamma_ - 1.) * internal_energy;
      }

    private:
      double gamma_;
      double b_;
      double q_;
      double pinf_;
    };
  } // namespace EquationOfStateLibrary
} /* namespace ryujin */
