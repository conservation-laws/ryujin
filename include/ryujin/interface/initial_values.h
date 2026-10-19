//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/linear_algebra/multicomponent_vector.h>

#include <deal.II/base/parameter_acceptor.h>

#include <string>

namespace ryujin
{
  namespace Interface
  {
    /**
     * The interface of InitialValues: it interpolates the initial (or an
     * analytic) hyperbolic state at a given time.
     *
     * @ingroup InitialValues
     */
    template <int dim, typename Number = double>
    class InitialValues : public dealii::ParameterAcceptor
    {
    public:
      /**
       * This routine computes and returns a hyperbolic state vector
       * populated with initial values for a specified time @p t.
       */
      virtual MultiComponentVector<Number>
      interpolate_hyperbolic_vector(Number t = 0) const = 0;

    protected:
      /**
       * Constructor.
       */
      explicit InitialValues(const std::string &subsection)
          : ParameterAcceptor(subsection)
      {
      }
    };
  } // namespace Interface
} /* namespace ryujin */
