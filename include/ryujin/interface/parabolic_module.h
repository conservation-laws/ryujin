//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/interface/module.h>

#include <ryujin/linear_algebra/state_vector.h>

#include <string>

namespace ryujin
{
  namespace Interface
  {
    /**
     * The description-free interface of a ParabolicModule: implicit time
     * stepping for a parabolic system.
     *
     * The interface holds the common state of Interface::Module.
     *
     * @ingroup ParabolicModule
     */
    template <int dim, typename Number = double>
    class ParabolicModule : public Module<dim, Number>
    {
    public:
      /**
       * @name Typedefs
       */
      //@{

      using typename Module<dim, Number>::StageVectors;

      using typename Module<dim, Number>::StageWeights;

      //@}
      /**
       * @name Functions for performing implicit time steps
       */
      //@{

      /**
       * Given a reference to a previous state vector @p old_state_vector
       * at time @p old_t and a time-step size @p tau perform an implicit
       * backward Euler step (and store the result in @p new_state_vector).
       *
       * The stage state vectors @p stage_state_vectors together with the
       * weights @p stage_weights are used to construct a modified high
       * order right-hand side / flux.
       *
       * @todo This function should always operate on the selected memory space.
       */
      virtual void
      backward_euler_step(const StateVector<Number> &old_state_vector,
                          const Number old_t,
                          StageVectors stage_state_vectors,
                          StageWeights stage_weights,
                          StateVector<Number> &new_state_vector,
                          Number tau) const = 0;

      /**
       * Given a reference to a previous state vector @p old_state_vector
       * at time @p old_t and a time-step size @p tau perform an implicit
       * Crank-Nicolson step (and store the result in @p new_state_vector).
       *
       * This variant is used in the TimeIntegrator class for the Strang
       * split variants.
       *
       * @todo This function should always operate on the selected memory space.
       */
      virtual void
      crank_nicolson_step(const StateVector<Number> &old_state_vector,
                          const Number old_t,
                          StateVector<Number> &new_state_vector,
                          Number tau) const = 0;

      //@}

    protected:
      using Module<dim, Number>::Module;
    };
  } // namespace Interface
} // namespace ryujin
