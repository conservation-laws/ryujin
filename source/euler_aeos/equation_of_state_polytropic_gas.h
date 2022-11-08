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
  } // namespace EquationOfStateLibrary
} /* namespace ryujin */
