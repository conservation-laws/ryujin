//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 by the ryujin authors
//

#pragma once

#include "equation_of_state.h"

#include <deal.II/base/function_parser.h>

namespace ryujin
{
  namespace EquationOfStateLibrary
  {
    /**
     * A user-specified equation of state
     *
     * @ingroup EulerEquations
     */
    class Function : public EquationOfState
    {
    public:
      using EquationOfState::pressure;
      using EquationOfState::specific_internal_energy;
      using EquationOfState::speed_of_sound;
      using EquationOfState::temperature;

      Function(const std::string &subsection)
          : EquationOfState("function", subsection)
      {
        p_expression_ = "(1.4 - 1.0) * rho * e";
        add_parameter(
            "pressure",
            p_expression_,
            "A function expression for the pressure as a function of density, "
            "rho, and specific internal energy, e: p(rho, e)");

        sie_expression_ = "p / (rho * (1.4 - 1.0))";
        add_parameter(
            "specific internal energy",
            sie_expression_,
            "A function expression for the specific internal energy as a "
            "function of density, rho, and pressure, p: e(rho, p)");

        temperature_expression_ = "e / 718.";
        add_parameter("temperature",
                      temperature_expression_,
                      "A function expression for the temperature as a "
                      "function of density, rho, and specific internal energy, "
                      "e: T(rho, e)");

        sos_expression_ = "sqrt(1.4 * (1.4 - 1.0) * e)";
        add_parameter(
            "speed of sound",
            sos_expression_,
            "A function expression for the speed of sound as a function of "
            "density, rho, and specific internal energy, e: s(rho, e)");

        add_parameter(
            "interpolatory covolume b",
            this->interpolation_b_,
            "The interpolatory maximum compressibility constant b used when "
            "constructing the interpolatory equation of state");

        add_parameter("interpolatory reference pressure",
                      this->interpolation_pinfty_,
                      "The interpolatory reference pressure p_infty used when "
                      "constructing the interpolatory equation of state");

        add_parameter(
            "interpolatory reference specific internal energy",
            this->interpolation_q_,
            "The interpolatory reference specific internal energy q used when "
            "constructing the interpolatory equation of state");

        /*
         * Set up the muparser object with the final equation of state
         * description from the parameter file:
         */
        const auto set_up_muparser = [this] {
          p_function_ = std::make_unique<dealii::FunctionParser<2>>();
          p_function_->initialize("rho,e", p_expression_, {});

          sie_function_ = std::make_unique<dealii::FunctionParser<2>>();
          sie_function_->initialize("rho,p", sie_expression_, {});

          temperature_function_ = std::make_unique<dealii::FunctionParser<2>>();
          temperature_function_->initialize(
              "rho,e", temperature_expression_, {});

          sos_function_ = std::make_unique<dealii::FunctionParser<2>>();
          sos_function_->initialize("rho,e", sos_expression_, {});
        };

        set_up_muparser();
        ParameterAcceptor::parse_parameters_call_back.connect(set_up_muparser);
      }

      double pressure(double rho, double e) const final
      {
        return p_function_->value(dealii::Point<2>(rho, e));
      }

      double specific_internal_energy(double rho, double p) const final
      {
        return sie_function_->value(dealii::Point<2>(rho, p));
      }

      double temperature(double rho, double e) const final
      {
        return temperature_function_->value(dealii::Point<2>(rho, e));
      }

      double speed_of_sound(double rho, double e) const final
      {
        return sos_function_->value(dealii::Point<2>(rho, e));
      }

    private:
      std::string p_expression_;
      std::string sie_expression_;
      std::string sos_expression_;
      std::string temperature_expression_;

      std::unique_ptr<dealii::FunctionParser<2>> p_function_;
      std::unique_ptr<dealii::FunctionParser<2>> sie_function_;
      std::unique_ptr<dealii::FunctionParser<2>> sos_function_;
      std::unique_ptr<dealii::FunctionParser<2>> temperature_function_;
    };
  } // namespace EquationOfStateLibrary
} // namespace ryujin
