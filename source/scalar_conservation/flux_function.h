//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "flux.h"

#include <deal.II/base/function_parser.h>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

#include <numeric>

namespace ryujin
{
  namespace FluxLibrary
  {
    /**
     * A generic flux description parsed from a user-provided string
     *
     * @ingroup ScalarConservation
     */
    class Function : public Flux
    {
    public:
      Function(const std::string &subsection)
          : Flux("function", subsection)
      {
        description_ = "0.5*u*u";
        add_parameter("description",
                      description_,
                      "A mathematical description of the flux as a function of "
                      "state used to create a muparser object to evaluate the "
                      "flux. For two, or three dimensional fluxes, components "
                      "are separated with a semicolon (;).");

        derivative_approximation_delta_ = 1.0e-10;
        add_parameter("derivative approximation delta",
                      derivative_approximation_delta_,
                      "Step size of the central difference quotient to compute "
                      "an approximation of the flux derivative");

        /*
         * Set up the muparser object with the final flux description from
         * the parameter file:
         */
        const auto set_up_muparser = [this] {

          std::vector<std::string> expression;
          boost::split(expression, description_, boost::is_any_of(";"));

          const auto size = expression.size();

          Assert(0 < size && size <= 3,
                 dealii::ExcMessage(
                     "user specified flux description must be either one, two, "
                     "or three strings separated by a comma"));
          flux_function_ = std::make_unique<dealii::FunctionParser<1>>(
              size, 0.0, derivative_approximation_delta_);
          flux_function_->initialize({"u"}, expression, {});

          flux_formula_ = "f(u)={" + description_ + "}";
        };

        set_up_muparser();
        ParameterAcceptor::parse_parameters_call_back.connect(set_up_muparser);
      }


      double value(const double state,
                   const unsigned int direction) const override
      {
        return flux_function_->value(dealii::Point<1>(state), direction);
      }


      double gradient(const double state,
                      const unsigned int direction) const override
      {
        return flux_function_->gradient(dealii::Point<1>(state), direction)[0];
      }


    private:
      std::string description_;
      double derivative_approximation_delta_;

      std::unique_ptr<dealii::FunctionParser<1>> flux_function_;
    };
  } // namespace EquationOfStateLibrary
} // namespace ryujin
