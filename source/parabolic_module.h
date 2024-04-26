//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2024 by the ryujin authors
//

#pragma once

#include "hyperbolic_module.h"

#include <compile_time_options.h>

#include "convenience_macros.h"
#include "initial_values.h"
#include "offline_data.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/sparse_matrix.templates.h>
#include <deal.II/lac/vector.h>

#include <functional>

namespace ryujin
{
  /**
   * Implicit backward Euler time-stepping and Crank-Nicolson time-stepping
   * for the parabolic subsystem.
   *
   * @ingroup ParabolicModule
   */
  template <typename Description, int dim, typename Number = double>
  class ParabolicModule final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @name Typedefs and constexpr constants
     */
    //@{

    using HyperbolicSystem = Description::HyperbolicSystem;

    using View = Description::template HyperbolicSystemView<dim, Number>;

    using ParabolicSystem = Description::ParabolicSystem;

    using ParabolicSolver = Description::template ParabolicSolver<dim, Number>;

    using StateVector = View::StateVector;

    //@}
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor.
     */
    ParabolicModule(
        const MPI_Comm &mpi_communicator,
        std::map<std::string, dealii::Timer> &computing_timer,
        const OfflineData<dim, Number> &offline_data,
        const HyperbolicSystem &hyperbolic_system,
        const ParabolicSystem &parabolic_system,
        const InitialValues<Description, dim, Number> &initial_values,
        const std::string &subsection = "/ParabolicModule");

    /**
     * Prepare time stepping. A call to @p prepare() allocates temporary
     * storage and is necessary before any of the following time-stepping
     * functions can be called.
     */
    void prepare();

    //@}
    /**
     * @name Functons for performing explicit time steps
     */
    //@{

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
    void step(const StateVector &old_state_vector,
              const Number old_t,
              std::array<std::reference_wrapper<const StateVector>, stages>
                  stage_state_vectors,
              const std::array<Number, stages> stage_weights,
              StateVector &new_state_vector,
              Number tau) const;

    /**
     * Print a status line with solver statistics. This function is used
     * for constructing the status message displayed periodically in the
     * TimeLoop.
     */
    void print_solver_statistics(std::ostream &output) const;

    //@}
    /**
     * @name Accessors
     */
    //@{
    /**
     * The number of restarts issued by the step() function.
     */
    ACCESSOR_READ_ONLY(n_restarts)

    /**
     * The number of ID violation warnings encounterd in the step()
     * function.
     */
    ACCESSOR_READ_ONLY(n_warnings)

    // FIXME: refactor to function
    mutable IDViolationStrategy id_violation_strategy_;

  private:
    //@}
    /**
     * @name Internal data
     */
    //@{

    // FIXME: refactor contents into this class.
    ParabolicSolver parabolic_solver_;
    mutable unsigned int cycle_;

    mutable unsigned int n_restarts_;
    mutable unsigned int n_warnings_;

    //@}
  };

} /* namespace ryujin */
