//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"
#include <initial_state_library.h>

#include <deal.II/base/function_parser.h>

namespace ryujin
{
  namespace ScalarConservation
  {
    struct Description;

    /**
     * Initial state defined by a user provided function
     *
     * @ingroup ScalarConservationEquations
     */
    template <int dim, typename Number>
    class Function : public InitialState<Description, dim, Number>
    {
    public:
      using View = HyperbolicSystemView<dim, Number>;
      using state_type = typename View::state_type;
      using primitive_state_type = typename View::primitive_state_type;

      Function(const HyperbolicSystem &hyperbolic_system,
               const std::string subsection)
          : InitialState<Description, dim, Number>("function", subsection)
          , hyperbolic_system(hyperbolic_system)
      {
        expression_ = "0.25 * x";
        this->add_parameter("expression",
                            expression_,
                            "A function expression for the initial state");

        /*
         * Set up the muparser object with the final flux description from
         * the parameter file:
         */
        const auto set_up_muparser = [this] {
          /*
           * This variant of the constructor initializes the function
           * parser with support for a time-dependent description involving
           * a variable »t«:
           */
          function_ =
              std::make_unique<dealii::FunctionParser<dim>>(expression_);
        };

        set_up_muparser();
        this->parse_parameters_call_back.connect(set_up_muparser);
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        function_->set_time(t);
        state_type result;
        result[0] = function_->value(point);
        return result;
      }

    private:
      const HyperbolicSystem &hyperbolic_system;

      std::string expression_;
      std::unique_ptr<dealii::FunctionParser<dim>> function_;
    };
  } // namespace ScalarConservation
} // namespace ryujin
