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
         * Derived EOS can override the covolume constant b.
         */
        covolume_constant_ = 0.;

        /*
         * Derived EOS can override the reference pressure that is used in
         * the interpolatory NASG approach.
         */
        interpolation_pinfty_ = 0.;

        /*
         * Derived EOS can override the reference specific internal energy
         * q that is used in the interpolatory NASG approach.
         */
        interpolation_q_ = 0.;
      }

      /**
       * Return the pressure given density @p rho and specific internal
       * energy @p e.
       *
       * @note This function is implemented for every equation of state.
       */
      virtual double pressure(double rho, double e) const = 0;

      /**
       * Return the specific internal energy @p e for a given density @p
       * rho and pressure @p p.
       *
       * @note This function is implemented for every equation of state.
       */
      virtual double specific_internal_energy(double rho, double p) const = 0;

      /**
       * Return the specific entropy @p s for a given density @p rho and
       * specific internal energy @p e.
       *
       * @note This function might not be implemented for a given equation
       * of state.
       */
      virtual double specific_entropy(double /*rho*/, double /*e*/) const
      {
        AssertThrow(false, dealii::ExcNotImplemented());
        __builtin_trap();
        return 0;
      }

      /**
       * Return the cold curve bound \f$e\ge e_{s_0}(\rho)\f$ that defines
       * the admissible set.
       *
       * @note This function might not be implemented for a given equation
       * of state.
       */
      virtual double cold_curve_bound(double /*rho*/) const
      {
        AssertThrow(false, dealii::ExcNotImplemented());
        __builtin_trap();
        return 0;
      }

      /**
       * Return the temperature @p T for a given density @p
       * rho and specific internal energy @p e.
       *
       * @note This function might not be implemented for a given equation
       * of state.
       */
      virtual double temperature(double /*rho*/, double /*e*/) const
      {
        AssertThrow(false, dealii::ExcNotImplemented());
        __builtin_trap();
        return 0;
      }

      /**
       * Return the sound speed @p c for a given density @p rho and
       * specific internal energy @p e.
       *
       * @note This function might not be implemented for a given equation
       * of state.
       */
      virtual double speed_of_sound(double /*rho*/, double /*e*/) const
      {
        AssertThrow(false, dealii::ExcNotImplemented());
        __builtin_trap();
        return 0;
      }

      /**
       * Return the covolume constant b.
       */
      ACCESSOR_READ_ONLY(covolume_constant)

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
      double covolume_constant_;

      double interpolation_pinfty_;
      double interpolation_q_;

    private:
      const std::string name_;
    };

  } // namespace EquationOfStateLibrary
} /* namespace ryujin */
