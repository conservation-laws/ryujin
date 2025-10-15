//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <hyperbolic_module.h>
#include <initial_values.h>
#include <mpi_ensemble.h>
#include <offline_data.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>

namespace ryujin
{
  /**
   * A stub parabolic solver that models the identity.
   *
   * @ingroup ParabolicModule
   */
  template <typename Description, int dim, typename Number>
  class StubParabolicModule final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @name Typedefs and constexpr constants
     */
    //@{

    using HyperbolicSystem = typename Description::HyperbolicSystem;

    using View =
        typename Description::template HyperbolicSystemView<dim, Number>;

    using ParabolicSystem = typename Description::ParabolicSystem;

    using StateVector = typename View::StateVector;

    //@}
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor.
     */
    StubParabolicModule(
        const MPIEnsemble & /*mpi_ensemle*/,
        std::map<std::string, dealii::Timer> & /*computing_timer*/,
        const OfflineData<dim, Number> & /*offline_data*/,
        const HyperbolicSystem & /*hyperbolic_system*/,
        const ParabolicSystem & /*parabolic_system*/,
        const InitialValues<Description, dim, Number> & /*initial_values*/,
        const std::string &subsection = "StubParabolicModule")
        : ParameterAcceptor(subsection)
    {
    }

    /**
     * Prepare time stepping. A call to @p prepare() allocates temporary
     * storage and is necessary before any of the following time-stepping
     * functions can be called.
     */
    void prepare()
    {
      // do nothing
    }

    //@}
    /**
     * @name Functons for performing explicit time steps
     */
    //@{

    void set_id_violation_strategy(
        const IDViolationStrategy & /*new_strategy*/) const
    {
      // do nothing
    }

    /**
     * This function preprocesses a given state vector @p U in preparation
     * for a high order IMEX time step. This function exists because some
     * time stepping variants have to precompute quantities before we can
     * perform an IMEX step. In addition, this function is called whenever
     * we perform a mesh transfer or output operation.
     */
    void prepare_state_vector(StateVector & /*state_vector*/,
                              Number /*t*/) const
    {
      Assert(false,
             dealii::ExcMessage("The parabolic system is the identity. This "
                                "function should have never been called."));
    }

    /**
     * Given a reference to a previous state vector @p old_U at time
     * @p old_t and a time-step size @p tau perform an implicit backward
     * euler step (and store the result in @p new_U).
     *
     * The function takes an optional array of states @p stage_U together
     * with a an array of weights @p stage_weights to construct a modified
     * high-order right-hand side / flux.
     */
    template <int stages>
    void
    backward_euler_step(const StateVector &old_state_vector,
                        const Number /*old_t*/,
                        std::array<std::reference_wrapper<const StateVector>,
                                   stages> /*stage_state_vectors*/,
                        const std::array<Number, stages> /*stage_weights*/,
                        StateVector &new_state_vector,
                        Number /*tau*/) const
    {
      Assert(false,
             dealii::ExcMessage("The parabolic system is the identity. This "
                                "function should have never been called."));

      new_state_vector = old_state_vector;
    }

    /**
     * Given a reference to a previous state vector @p old_U at time @p
     * old_t and a time-step size @p tau perform an implicit Crank-Nicolson
     * step (and store the result in @p new_U).
     *
     * This variant is used in the TimeIntegrator class for the Strang
     * split variants.
     */
    void crank_nicolson_step(const StateVector &old_state_vector,
                             const Number /*old_t*/,
                             StateVector &new_state_vector,
                             Number /*tau*/) const
    {
      Assert(false,
             dealii::ExcMessage("The parabolic system is the identity. This "
                                "function should have never been called."));

      new_state_vector = old_state_vector;
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
    void print_solver_statistics(std::ostream & /*output*/) const
    {
      // do nothing
    }

    /**
     * The number of restarts signalled by the step() function.
     */
    unsigned int n_restarts() const
    {
      return 0;
    }

    /**
     * The number of corrections performed by the step() function. This
     * function exists to mirror the ParabolicModule interface and will
     * always return 0.
     */
    unsigned int n_corrections() const
    {
      return 0;
    }

    /**
     * The number of ID violation warnings encounterd in the step()
     * function.
     */
    unsigned int n_warnings() const
    {
      return 0;
    }

    //@}
  };
} /* namespace ryujin */
