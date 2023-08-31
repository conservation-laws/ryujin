//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_acceptor.h>

#include <string>

namespace ryujin
{
  namespace FluxLibrary
  {
    /**
     * A small abstract base class to group configuration options for the
     * flux.
     *
     * This function derives directly from dealii::Function<1>. Derived
     * classes must thus implement the value() and gradient() methods.
     *
     * @ingroup ScalarConservation
     */
    class Flux : public dealii::ParameterAcceptor
    {
    public:
      /**
       * Constructor taking flux name @p name and a subsection @p
       * subsection as an argument. The dealii::ParameterAcceptor is
       * initialized with the subsubsection `subsection + "/" + name`.
       */
      Flux(const std::string &name, const std::string &subsection)
          /* simply default to three components... */
          : ParameterAcceptor(subsection + "/" + name)
          , name_(name)
      {
      }


      /**
       * Return the flux f(u) for the given state @p U and direction
       * @p direction.
       */
      virtual double value(double state, unsigned int direction) const = 0;


      /**
       * Return the gradient f'(u) of the flux for the given state @p u and
       * direction @p direction.
       */
      virtual double gradient(double state, unsigned int direction) const = 0;

      /**
       * The name of the flux function
       */
      ACCESSOR_READ_ONLY(name);

      /**
       * A string showing a detailed formula of the chosen flux function
       * (such as "f(u)=0.5*u*u").
       */
      ACCESSOR_READ_ONLY(flux_formula);

    protected:
      std::string flux_formula_;

    private:
      const std::string name_;
    };

  } // namespace FluxLibrary
} /* namespace ryujin */
