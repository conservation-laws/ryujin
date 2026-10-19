//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"

#include <deal.II/base/array_view.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <string>

namespace ryujin
{
  namespace BarotropicEquationOfStateLibrary
  {
    /**
     * A small abstract base class to group configuration options for a
     * barotropic equation of state.
     *
     * @ingroup EulerEquations
     */
    class BarotropicEquationOfState : public dealii::ParameterAcceptor
    {
    public:
      /**
       * Constructor taking EOS name @p name and a subsection @p subsection
       * as an argument. The dealii::ParameterAcceptor is initialized with
       * the subsubsection `subsection + "/" + name`.
       */
      BarotropicEquationOfState(const std::string &name,
                                const std::string &subsection)
          : ParameterAcceptor(subsection + "/" + name)
          , name_(name)
      {
      }

      /**
       * Return the specific internal energy @p e for a given density @p rho.
       */
      virtual double specific_internal_energy(double rho) const = 0;

      /**
       * Return the pressure for a given density @p rho.
       */
      virtual double pressure(double rho) const = 0;

      /**
       * Return the sound speed @p c for a given density @p rho.
       */
      virtual double speed_of_sound(double rho) const = 0;

      /**
       * Return the name of the EOS as (const reference) std::string
       */
      ACCESSOR_READ_ONLY(name)

    private:
      const std::string name_;
    };

  } // namespace BarotropicEquationOfStateLibrary
} /* namespace ryujin */
