//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <string>

namespace ryujin
{
  namespace EulerAEOS
  {
    /**
     * A small abstract base class to group configuration options for an
     * equation of state.
     *
     * @ingroup EulerEquations
     */
    class EquationOfState : public dealii::ParameterAcceptor
    {
    public:
      /**
       * Constructor taking EOS name @p name and a subsection @p subsection
       * as an argument. The dealii::ParameterAcceptor is initialized with
       * the subsubsection `subsection + "/" + name`.
       */
      EquationOfState(const std::string &name, const std::string &subsection)
          : ParameterAcceptor(subsection + "/" + name)
          , name_(name)
      {
        /*
         * If necessary derived EOS can override the inerpolation co-volume
         * b that is used in the approximate Riemann solver.
         */
        interpolation_b_ = 0.;
      }

      /**
       * Return the pressure given density @p rho and specific internal
       * energy @p e.
       */
      virtual double pressure(const double &rho, const double &e) = 0;

      /**
       * Return the specific internal energy @p e for a given density @p
       * rho and pressure @p p.
       */
      virtual double specific_internal_energy(const double &rho,
                                              const double &p) = 0;

      /**
       * Return the sound speed @p c for a given density @p rho and
       * specific internal energy  @p e.
       */
      virtual double sound_speed(const double &rho, const double &e) = 0;

      /**
       * Return the interpolation co-volume constant (b).
       */
      ACCESSOR_READ_ONLY(interpolation_b)

      /**
       * Return the name of the EOS as (const reference) std::string
       */
      ACCESSOR_READ_ONLY(name)

    protected:
      double interpolation_b_;

    private:
      const std::string name_;
    };

  } // namespace EulerAEOS
} /* namespace ryujin */
