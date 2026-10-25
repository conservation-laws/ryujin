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
     * The description-free interface of a TimeIntegrator: it advances the
     * state vector of a Simulation by one time step.
     *
     * In addition to the time step the interface holds the name and the
     * efficiency of the time-stepping scheme.
     *
     * @ingroup Interfaces
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
       * Prepare time integration. A call to prepare() is necessary before
       * any of the following functions can be called.
       */
      virtual void prepare() = 0;

      //@}
      /**
       * @name Functions for performing time steps
       */
      //@{

      /**
       * Given a reference to a previous state vector @p state_vector at
       * time @p t perform a time step (and store the result in @p
       * state_vector). The function returns the time step size tau with
       * which the update was performed.
       *
       * If @p tau is nonzero the step is performed with @p tau. If @p tau
       * is 0 the time step size is chosen such that it is less than or
       * equal to @p tau_max.
       *
       * @pre The @p state_vector has to be prepared with
       * Simulation::prepare_state_vector() prior to calling step().
       */
      virtual Number
      step(StateVector<Number> &state_vector,
           Number t,
           Number tau,
           Number tau_max = std::numeric_limits<Number>::max()) = 0;

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

      std::string time_stepping_scheme_name_;
      Number efficiency_;
    };
  } // namespace Interface
} // namespace ryujin
