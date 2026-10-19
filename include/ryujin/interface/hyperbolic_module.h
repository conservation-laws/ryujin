//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/interface/module.h>

#include <ryujin/base/convenience_macros.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <deal.II/base/exceptions.h>

#include <limits>
#include <string>

namespace ryujin
{
  namespace Interface
  {
    /**
     * The description-free interface of a HyperbolicModule: explicit
     * forward Euler time-stepping for a hyperbolic system with convex
     * limiting.
     *
     * In addition to the common state of Interface::Module the interface
     * holds the CFL number and the acceptable tau_max ratio.
     *
     * @ingroup HyperbolicModule
     */
    template <int dim, typename Number = double>
    class HyperbolicModule : public Module<dim, Number>
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
       * @name Functions for performing explicit time steps
       */
      //@{

      /**
       * Populate the precomputed values of the state vector @p state_vector
       * from its hyperbolic state and update the ghost values. In contrast
       * to prepare_state_vector() no boundary conditions are applied and
       * constrained degrees of freedom are not skipped.
       *
       * @note This function operates on the selected memory space.
       */
      virtual void
      fill_precomputed_values(StateVector<Number> &state_vector) const = 0;

      /**
       * Given a reference to a previous state vector @p old_state_vector
       * perform an explicit Euler step (and store the result in @p
       * new_state_vector). The function returns the chosen time step size
       * tau with which the update was performed.
       *
       * The time step is performed with either tau_max (if @p tau is set
       * to 0), or tau (if @p tau is nonzero). Here, tau_max is the minimum
       * of the specified parameter @p tau_max and the computed maximal time
       * step size according to the CFL condition.
       *
       * The stage state vectors @p stage_state_vectors together with the
       * weights @p stage_weights are used to construct a modified high
       * order flux, see the typed HyperbolicModule for details.
       *
       * @note The state vector @p old_state_vector has to be prepared
       * with prepare_state_vector() prior to calling the step function.
       *
       * @note This function operates on the selected memory space.
       */
      virtual Number
      step(const StateVector<Number> &old_state_vector,
           StageVectors stage_state_vectors,
           StageWeights stage_weights,
           StateVector<Number> &new_state_vector,
           Number tau = Number(0.),
           Number tau_max = std::numeric_limits<Number>::max()) const = 0;

      /**
       * Sets the relative CFL number used for computing an appropriate
       * time-step size to the given value. The CFL number must be a
       * positive value. If chosen to be within the interval \f$(0,1)\f$
       * then the low-order update and limiting stages guarantee invariant
       * domain preservation.
       */
      void set_cfl(Number new_cfl) const
      {
        Assert(new_cfl > Number(0.), dealii::ExcInternalError());
        cfl_ = new_cfl;
      }

      /**
       * Sets the relative maximal acceptable tau_max ratio. If the ratio
       * of the enforced time-step size tau and our computed tau_max for
       * the hyperbolic (sub) step is above this limit we throw a Restart
       * exception.
       */
      void
      set_acceptable_tau_max_ratio(Number new_acceptable_tau_max_ratio) const
      {
        Assert(new_acceptable_tau_max_ratio >= Number(1.),
               dealii::ExcInternalError());
        acceptable_tau_max_ratio_ = new_acceptable_tau_max_ratio;
      }

      //@}
      /**
       * @name Information and statistics
       */
      //@{

      /**
       * Returns the relative CFL number used for computing an appropriate
       * time-step size.
       */
      ACCESSOR_READ_ONLY(cfl)

      /**
       * Returns the relative maximal acceptable tau_max ratio.
       */
      ACCESSOR_READ_ONLY(acceptable_tau_max_ratio)

      /**
       * The number of components of the hyperbolic state, a run time copy
       * of the compile time constant of the hyperbolic system.
       */
      ACCESSOR_READ_ONLY(problem_dimension)

      /**
       * The number of precomputed values, a run time copy of the compile
       * time constant of the hyperbolic system.
       */
      ACCESSOR_READ_ONLY(n_precomputed_values)

      //@}

    protected:
      /**
       * Constructor.
       */
      HyperbolicModule(const unsigned int problem_dimension,
                       const unsigned int n_precomputed_values,
                       const std::string &subsection)
          : Module<dim, Number>(subsection)
          , cfl_(Number(0.2))
          , acceptable_tau_max_ratio_(Number(1.e6))
          , problem_dimension_(problem_dimension)
          , n_precomputed_values_(n_precomputed_values)
      {
      }

      mutable Number cfl_;
      mutable Number acceptable_tau_max_ratio_;

    private:
      const unsigned int problem_dimension_;
      const unsigned int n_precomputed_values_;
    };
  } // namespace Interface
} // namespace ryujin
