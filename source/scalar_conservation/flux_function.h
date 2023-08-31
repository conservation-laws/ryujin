//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "flux.h"

#include <deal.II/base/function_parser.h>

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
        description_ = {"0.5 * u * u"};
        add_parameter("description",
                      description_,
                      "A mathematical description of the flux as a function of "
                      "state used to create a muparser object to evaluate the "
                      "flux. For two, or three dimensional fluxes, components "
                      "are separated with a comma (,).");

        /*
         * Set up the muparser object with the final flux description from
         * the parameter file:
         */
        const auto set_up_muparser = [this] {
          const auto size = description_.size();
          Assert(0 < size && size <= 3,
                 dealii::ExcMessage(
                     "user specified flux description must be either one, two, "
                     "or three strings separated by a comma"));
          flux_function_ = std::make_unique<dealii::FunctionParser<1>>(
              size, 0.0, derivative_approximation_delta_);
          flux_function_->initialize({"u"}, description_, {});

          flux_formula_ =
              "f(u)={" +
              std::accumulate(std::begin(description_),
                              std::end(description_),
                              std::string(),
                              [](std::string &result, std::string &element) {
                                return result.empty() ? element
                                                      : result + "," + element;
                              }) +
              "}";
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
      std::vector<std::string> description_;
      double derivative_approximation_delta_;

      std::unique_ptr<dealii::FunctionParser<1>> flux_function_;
    };
  } // namespace EquationOfStateLibrary
} // namespace ryujin
