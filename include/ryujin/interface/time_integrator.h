//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/convenience_macros.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <deal.II/base/parameter_acceptor.h>

#include <limits>
#include <string>

namespace ryujin
{
  namespace Interface
  {
    /**
     * The interface of a TimeIntegrator: a time stepping scheme that
     * advances a state vector by combining the step functions of a
     * hyperbolic and a parabolic module.
     *
     * @ingroup TimeLoop
     */
    template <int dim, typename Number = double>
    class TimeIntegrator : public dealii::ParameterAcceptor
    {
    public:
      /**
       * @name Setup
       */
      //@{

      /**
       * Prepare time integration. A call to prepare() allocates temporary
       * storage and is necessary before any of the following
       * time-stepping functions can be called.
       */
      virtual void prepare() = 0;

      //@}
      /**
       * @name Functions for performing time steps
       */
      //@{

      /**
       * This function preprocesses a given state vector for time
       * stepping. It has to be called prior to step() to ensure that
       * precomputed values are in place.
       */
      virtual void prepare_state_vector(StateVector<Number> &state_vector,
                                        Number t) const = 0;

      /**
       * Given a reference to a previous state vector performs a time step
       * (and store the result in @p state_vector). The function returns
       * the chosen time step size tau. The time step size tau is selected
       * such that $t + tau <= t_final$.
       *
       * @pre The @p state_vector has to be prepared with the
       * prepare_state_vector() function prior to calling the step()
       * function.
       */
      virtual Number
      step(StateVector<Number> &state_vector,
           Number t,
           Number t_final = std::numeric_limits<Number>::max()) = 0;

      //@}
      /**
       * @name Information and statistics
       */
      //@{

      /**
       * The name of the selected time-stepping scheme.
       *
       * @note Valid once prepare() has been called.
       */
      ACCESSOR_READ_ONLY(time_stepping_scheme_name)

      /**
       * The efficiency of the selected time-stepping scheme expressed as
       * the ratio of step size of the combined method to step size of an
       * elementary forward Euler step. For example, SSPRK33 has an
       * efficiency ratio of 1 whereas ERK33 has an efficiency ratio of 3.
       *
       * @note Valid once prepare() has been called.
       */
      ACCESSOR_READ_ONLY(efficiency)

      //@}

    protected:
      /**
       * Constructor.
       */
      explicit TimeIntegrator(const std::string &subsection)
          : ParameterAcceptor(subsection)
          , efficiency_(Number(1.))
      {
      }

      /**
       * The name of the selected time-stepping scheme.
       *
       * @note A derived class sets it in prepare().
       */
      std::string time_stepping_scheme_name_;

      /**
       * The efficiency of the selected time-stepping scheme.
       *
       * @note A derived class sets it in prepare().
       */
      Number efficiency_;
    };
  } // namespace Interface
} // namespace ryujin
