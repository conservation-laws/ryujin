//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <deal.II/base/parameter_acceptor.h>

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <ostream>
#include <string>

namespace ryujin
{
  /**
   * An enum controlling the behavior on detection of an invariant domain
   * or CFL violation. Such a case might occur for either aggressive CFL
   * numbers > 1, and/or later stages in the Runge Kutta scheme when the
   * time step tau is prescribed.
   *
   * The invariant domain violation is detected in the limiter and implies
   * that the low-order update itself lies outside the invariant domain.
   *
   * A "CFL violation" occurs if the computed tau_max of a substep is
   * significantly larger than the previously computed time step size tau
   * with which we perform the update.
   *
   * @note Data structures in HyperbolicModule are initialized with the
   * ensemble subrange communicator stored in MPIEnsemble. However, the
   * time step size constraint (i.e. tau_max) is synchronized over the
   * entire global communicator.
   *
   * @ingroup Interfaces
   */
  enum class IDViolationStrategy : std::uint8_t {
    /**
     * Warn about an invariant domain violation but take no further
     * action.
     */
    warn,

    /**
     * Raise a Restart exception on domain violation. This exception can be
     * caught in TimeIntegrator and various different actions (adapt CFL
     * and retry) can be taken depending on chosen strategy.
     */
    raise_exception,
  };


  /**
   * A class signalling a restart, thrown in HyperbolicModule::single_step
   * and caught in the TimeIntegrator when performing a time step. We
   * currently signal a restart whenever we encounter an invariant domain
   * violation of the low-order update.
   *
   * This might happen, for example, in subsequent invocations of step() in
   * high-order time stepping algorithms when the time step size has to be
   * kept constant. A standard strategy implemented in TimeIntegrator is to
   * simply fall back to a smaller relative CFL number and try again.
   *
   * The field @p suggested_tau_max is set to a tau_max value with which
   * the entire update step should be restarted.
   *
   * @ingroup Interfaces
   */
  class Restart final
  {
  public:
    double suggested_tau_max;
  };


  /**
   * A class signalling that a correction step is required for the update.
   * This is meant to be a truly exceptional situation, for example where
   * the high-order update violates an admissibility condition. With our
   * limiting approach this must never happen, ... but we might have taken
   * some short cuts in the parabolic update that under rare circumstances
   * require limiting.
   *
   * @ingroup Interfaces
   */
  class Correction final
  {
  };


  namespace Interface
  {
    /**
     * The common description-free part of a HyperbolicModule and a
     * ParabolicModule: the setup of the module and of the parts of the
     * state vector that its system owns, the invariant domain violation
     * strategy, and the statistics of the step functions.
     *
     * The step functions differ between the explicit hyperbolic and the
     * implicit parabolic update and are declared in
     * Interface::HyperbolicModule and Interface::ParabolicModule.
     *
     * @ingroup Interfaces
     */
    template <int dim, typename Number = double>
    class Module : public dealii::ParameterAcceptor
    {
    public:
      /**
       * @name Typedefs
       */
      //@{

      /**
       * The stage state vectors of a step function.
       */
      using StageVectors = std::initializer_list<
          std::reference_wrapper<const StateVector<Number>>>;

      /**
       * The stage weights of a step function, see StageVectors.
       */
      using StageWeights = std::initializer_list<Number>;

      //@}
      /**
       * @name Setup and preparation of state vectors
       */
      //@{

      /**
       * Prepare time stepping. A call to @p prepare() allocates temporary
       * storage and is necessary before any of the step functions can be
       * called.
       *
       * @note This function operates on the selected memory space.
       */
      virtual void prepare() = 0;

      /**
       * (Re)initialize the parts of the state vector that the system of
       * the module owns: the state and precomputed parts for a
       * HyperbolicModule, the dynamic parts for a ParabolicModule.
       *
       * @note This routine does not modify the parts owned by the other
       * system.
       *
       * @note This function operates on the selected memory space.
       */
      virtual void
      reinit_state_vector(StateVector<Number> &state_vector) const = 0;

      /**
       * Preprocess a given state vector for the time @p t in preparation
       * for a step of the module, an output, or a mesh transfer.
       *
       * @note This function operates on the selected memory space.
       */
      virtual void prepare_state_vector(StateVector<Number> &state_vector,
                                        Number t) const = 0;

      /**
       * Sets the invariant domain violation strategy.
       */
      void set_id_violation_strategy(const IDViolationStrategy &strategy) const
      {
        id_violation_strategy_ = strategy;
      }

      //@}
      /**
       * @name Information and statistics
       */
      //@{

      /**
       * Print a status line with solver statistics. This function is used
       * for constructing the status message displayed periodically in the
       * TimeLoop.
       */
      virtual void print_solver_statistics(std::ostream &output) const = 0;

      /**
       * The number of restarts signalled by the step functions. We signal
       * a restart whenever we encounter an ID violation (and throwing a
       * restart is enabled).
       */
      ACCESSOR_READ_ONLY(n_restarts)

      /**
       * The number of corrections performed by the step functions.
       */
      ACCESSOR_READ_ONLY(n_corrections)

      /**
       * The number of ID violation warnings encountered in the step
       * functions. We issue a warning whenever we encounter an ID
       * violation (and throwing a restart is disabled).
       */
      ACCESSOR_READ_ONLY(n_warnings)

      //@}

    protected:
      /**
       * Constructor.
       */
      explicit Module(const std::string &subsection)
          : ParameterAcceptor(subsection)
          , id_violation_strategy_(IDViolationStrategy::warn)
          , n_restarts_(0)
          , n_corrections_(0)
          , n_warnings_(0)
      {
      }

      mutable IDViolationStrategy id_violation_strategy_;

      mutable unsigned int n_restarts_;
      mutable unsigned int n_corrections_;
      mutable unsigned int n_warnings_;
    };
  } // namespace Interface
} // namespace ryujin
