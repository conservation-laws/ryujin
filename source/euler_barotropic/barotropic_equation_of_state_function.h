//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "barotropic_equation_of_state.h"

#include <deal.II/base/function_parser.h>

namespace ryujin
{
  namespace BarotropicEquationOfStateLibrary
  {
    /**
     * A user-specified equation of state for the barotropic Euler equations.
     *
     * The specific internal energy, pressure and speed of sound have to
     * satisfy the following relationships:
     *
     * \f{align}
     *   p &= \rho^2 \,\partial_\rho e(\rho), \qquad
     *   a &= \sqrt{\partial_\rho p(\rho)}.
     * \f}
     *
     * @ingroup EulerEquations
     */
    class Function : public BarotropicEquationOfState
    {
    public:
      Function(const std::string &subsection)
          : BarotropicEquationOfState("function", subsection)
      {
        sie_expression_ = "4. * ln(rho)";
        add_parameter("specific internal energy",
                      sie_expression_,
                      "A function expression for the specific internal energy "
                      "as a function of density: e(rho)");

        p_expression_ = "4. * rho";
        add_parameter("pressure",
                      p_expression_,
                      "A function expression for the pressure as a function of "
                      "density: p(rho)");

        sos_expression_ = "2.";
        add_parameter("speed of sound",
                      sos_expression_,
                      "A function expression for the speed of sound as a "
                      "function of density: a(rho)");

        /*
         * Set up the muparser object with the final equation of state
         * description from the parameter file:
         */
        const auto set_up_muparser = [this] {
          sie_function_ = std::make_unique<dealii::FunctionParser<1>>();
          sie_function_->initialize("rho", sie_expression_, {});

          p_function_ = std::make_unique<dealii::FunctionParser<1>>();
          p_function_->initialize("rho", p_expression_, {});

          sos_function_ = std::make_unique<dealii::FunctionParser<1>>();
          sos_function_->initialize("rho", sos_expression_, {});
        };

        set_up_muparser();
        ParameterAcceptor::parse_parameters_call_back.connect(set_up_muparser);
      }

      double specific_internal_energy(double rho) const final
      {
        return sie_function_->value(dealii::Point<1>(rho));
      }

      double pressure(double rho) const final
      {
        return p_function_->value(dealii::Point<1>(rho));
      }

      double speed_of_sound(double rho) const final
      {
        return sos_function_->value(dealii::Point<1>(rho));
      }

    private:
      std::string sie_expression_;
      std::string p_expression_;
      std::string sos_expression_;

      std::unique_ptr<dealii::FunctionParser<1>> sie_function_;
      std::unique_ptr<dealii::FunctionParser<1>> p_function_;
      std::unique_ptr<dealii::FunctionParser<1>> sos_function_;
    };
  } // namespace BarotropicEquationOfStateLibrary
} // namespace ryujin
