//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <string>

namespace ryujin
{
  namespace EquationOfStateLibrary
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
         * If necessary derived EOS can override the covolume b that is
         * used in the interpolatory NASG eos.
         */
        interpolation_b_ = 0.;

        /*
         * If necessary derived EOS can override the reference pressure
         * that is used in the interpolatory NASG eos.
         */
        interpolation_pinfty_ = 0.;

        /*
         * If necessary derived EOS can override the reference specific
         * internal energy q that is used in the interpolatory NASG eos.
         */
        interpolation_q_ = 0.;
      }

      /**
       * Return the pressure given density @p rho and specific internal
       * energy @p e.
       */
      virtual double pressure(double rho, double e) const = 0;

      /**
       * Return the specific internal energy @p e for a given density @p
       * rho and pressure @p p.
       */
      virtual double specific_internal_energy(double rho, double p) const = 0;

      /**
       * Return the temperature @p T for a given density @p
       * rho and specific internal energy @p e.
       */
      virtual double temperature(double rho, double e) const = 0;

      /**
       * Return the sound speed @p c for a given density @p rho and
       * specific internal energy  @p e.
       */
      virtual double speed_of_sound(double rho, double e) const = 0;

      /**
       * Return the interpolation covolume constant (b).
       */
      ACCESSOR_READ_ONLY(interpolation_b)

      /**
       * Return the interpolation reference pressure (pinfty).
       */
      ACCESSOR_READ_ONLY(interpolation_pinfty)

      /**
       * Return the interpolation reference specific internal energy (q).
       */
      ACCESSOR_READ_ONLY(interpolation_q)

      /**
       * Return the name of the EOS as (const reference) std::string
       */
      ACCESSOR_READ_ONLY(name)

    protected:
      double interpolation_b_;
      double interpolation_pinfty_;
      double interpolation_q_;

    private:
      const std::string name_;
    };

  } // namespace EquationOfStateLibrary
} /* namespace ryujin */
