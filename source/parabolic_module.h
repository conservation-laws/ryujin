//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "hyperbolic_module.h"

#include <compile_time_options.h>

#include "convenience_macros.h"
#include "initial_values.h"
#include "offline_data.h"
#include "simd.h"
#include "sparse_matrix_simd.h"

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
     * @copydoc HyperbolicSystem
     */
    using HyperbolicSystem = typename Description::HyperbolicSystem;

    /**
     * @copydoc ParabolicSystem
     */
    using ParabolicSystem = typename Description::ParabolicSystem;

    /**
     * @copydoc HyperbolicSystem::View
     */
    using HyperbolicSystemView =
        typename HyperbolicSystem::template View<dim, Number>;

    /**
     * @copydoc HyperbolicSystem::View::vector_type
     */
    using vector_type = typename HyperbolicSystemView::vector_type;

    /**
     * Constructor.
     */
    ParabolicModule(
        const MPI_Comm &mpi_communicator,
        std::map<std::string, dealii::Timer> &computing_timer,
        const OfflineData<dim, Number> &offline_data,
        const ParabolicSystem &parabolic_system,
        const InitialValues<Description, dim, Number> &initial_values,
        const std::string &subsection = "/ParabolicModule");

    /**
     * Prepare time stepping. A call to @p prepare() allocates temporary
     * storage and is necessary before any of the following time-stepping
     * functions can be called.
     */
    void prepare();

    /**
     * @name Functons for performing explicit time steps
     */
    //@{

    /**
     * Given a reference to a previous state vector @p old_U and a
     * time-step size @p tau perform an implicit backward euler step (and
     * store the result in @p new_U).
     *
     * The function takes an optional array of states @p stage_U together
     * with a an array of weights @p stage_weights to construct a modified
     * high-order right-hand side / flux.
     */
    template <int stages>
    void
    step(const vector_type &old_U,
         std::array<std::reference_wrapper<const vector_type>, stages> stage_U,
         const std::array<Number, stages> stage_weights,
         vector_type &new_U,
         Number tau) const;

    /**
     * Given a reference to a previous state vector @p old_U and a
     * time-step size @p tau perform an implicit Crank-Nicolson step (and
     * store the result in @p new_U).
     */
    void crank_nicolson_step(const vector_type &old_U,
                             vector_type &new_U,
                             Number tau) const;

    //@}
    /**
     * @name Accessors
     */
    //@{

    /**
     * Return a reference to the OfflineData object
     */
    ACCESSOR_READ_ONLY(offline_data)

    /**
     * Return a reference to the HyperbolicSystem object
     */
    ACCESSOR_READ_ONLY(parabolic_system)

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

    const MPI_Comm &mpi_communicator_;
    std::map<std::string, dealii::Timer> &computing_timer_;

    dealii::SmartPointer<const OfflineData<dim, Number>> offline_data_;
    dealii::SmartPointer<const ParabolicSystem> parabolic_system_;
    dealii::SmartPointer<const InitialValues<Description, dim, Number>>
        initial_values_;

    mutable unsigned int n_restarts_;
    mutable unsigned int n_warnings_;

    //@}
  };

} /* namespace ryujin */
