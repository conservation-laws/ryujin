//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

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
     * The name of the parabolic system as a string.
     *
     * @todo Refactor into ParabolicSystem class
     */
    static inline const std::string problem_name = "Identity";

    /**
     * @copydoc HyperbolicSystem
     */
    using HyperbolicSystem = typename Description::HyperbolicSystem;

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
     * Typedef for a MultiComponentVector storing precomputed values.
     */
    using precomputed_vector_type =
        typename HyperbolicSystemView::precomputed_vector_type;


    /**
     * Constructor.
     */
    ParabolicModule(
        const MPI_Comm &mpi_communicator,
        std::map<std::string, dealii::Timer> &computing_timer,
        const OfflineData<dim, Number> &offline_data,
        const HyperbolicSystem &hyperbolic_system,
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
         const precomputed_vector_type &old_precomputed,
         std::array<std::reference_wrapper<const vector_type>, stages> stage_U,
         std::array<std::reference_wrapper<const precomputed_vector_type>,
                    stages> stage_precomputed,
         const std::array<Number, stages> stage_weights,
         vector_type &new_U,
         Number tau) const;

    /**
     * Given a reference to a previous state vector @p old_U and a
     * time-step size @p tau perform an implicit Crank-Nicolson step (and
     * store the result in @p new_U).
     */
    void crank_nicolson_step(const vector_type &old_U,
                             const precomputed_vector_type &old_precomputed,
                             vector_type &new_U,
                             Number tau) const;

    /**
     * Return a reference to the OfflineData object
     */
    ACCESSOR_READ_ONLY(offline_data)

    /**
     * Return a reference to the HyperbolicSystem object
     */
    ACCESSOR_READ_ONLY(hyperbolic_system)

    /**
     * The number of restarts issued by the step() function.
     */
    ACCESSOR_READ_ONLY(n_restarts)

    /**
     * The number of ID violation warnings encounterd in the step()
     * function.
     */
    ACCESSOR_READ_ONLY(n_warnings)

  private:
    //@}
    /**
     * @name Run time options
     */
    //@{

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPI_Comm &mpi_communicator_;
    std::map<std::string, dealii::Timer> &computing_timer_;

    dealii::SmartPointer<const OfflineData<dim, Number>> offline_data_;
    dealii::SmartPointer<const HyperbolicSystem> hyperbolic_system_;
    dealii::SmartPointer<const InitialValues<Description, dim, Number>>
        initial_values_;

    mutable unsigned int n_restarts_;
    mutable unsigned int n_warnings_;

    //@}
  };

} /* namespace ryujin */
