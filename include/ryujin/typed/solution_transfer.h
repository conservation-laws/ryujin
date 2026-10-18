//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception or LGPL-2.1-or-later
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <ryujin/interface/discretization.h>
#include <ryujin/interface/offline_data.h>
#include <ryujin/interface/parabolic_system.h>
#include <ryujin/interface/solution_transfer.h>

#include <mpi_ensemble.h>
#include <observer_pointer.h>
#include <state_vector.h>

#include <deal.II/base/parameter_acceptor.h>

#include <utility>
#include <vector>

namespace ryujin
{
  /**
   * The SolutionTransfer class is an adaptation of the
   * parallel::distributed::SolutionTransfer class and method implemented
   * in deal.II. This class is, in contrast to the deal.II version,
   * specifically tailored to our needs:
   *  - it uses the MultiComponentVector directly,
   *  - it implements a local mass matrix projection with convex limiting
   *    for the hyperbolic state,
   *  - it interpolates the dynamic state, without limiting, on the
   *    continuous finite element space of OfflineData::dof_handler_cg().
   *
   * @fixme generalize the transfer of the dynamic state to support
   * OfflineData::dof_handler_cg() and OfflineData::dof_handler_dg()
   *
   * @ingroup Mesh
   */
  template <typename HyperbolicDescription, int dim, typename Number = double>
  class SolutionTransfer final : public Interface::SolutionTransfer<dim, Number>
  {
  public:
    /**
     * @name Typedefs and constexpr constants
     */
    //@{

    using HyperbolicSystem = typename HyperbolicDescription::HyperbolicSystem;

    using View = typename HyperbolicSystem::template View<dim, Number>;

    using Limiter = typename HyperbolicDescription::template Limiter<Number>;

    using LimiterView =
        decltype(std::declval<const Limiter &>().template view<dim, Number>());

    static constexpr auto problem_dimension = View::problem_dimension;

    static constexpr auto n_precomputed_values = View::n_precomputed_values;

    using state_type = typename View::state_type;

    /**
     * The number of stored entries in the bounds array.
     */
    static constexpr unsigned int n_bounds = LimiterView::n_bounds;

    /**
     * Array type used to store accumulated bounds.
     */
    using Bounds = typename LimiterView::Bounds;

    //@}
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor.
     */
    SolutionTransfer(const MPIEnsemble &mpi_ensemble,
                     Interface::Discretization<dim> &discretization, /*mutable*/
                     const Interface::OfflineData<dim, Number> &offline_data,
                     const HyperbolicSystem &hyperbolic_system,
                     const Interface::ParabolicSystem &parabolic_system,
                     const std::string &subsection = "/SolutionTransfer");

    /**
     * Destructor
     */
    ~SolutionTransfer() override = default;

    //@}
    /**
     * @name Functions for solution transfer after mesh adaptation and for
     * serialization/deserialization
     */
    //@{

    /**
     * Prepare projection (and serialization) by registering a call back to
     * all cells of the distributed triangulation. The call back attaches
     * all necessary data to the triangulation to perform a projection
     * after mesh adaptation, or a Triangulation::store()/load() operation.
     */
    void
    prepare_projection(const StateVector<Number> &old_state_vector) override;

    /**
     * Project the data stored in the triangulation into a new state vector
     * @p new_state_vector. The attached data either comes from a previous
     * call to prepare_projection() prior to mesh adaptation or has been
     * read back in from a checkpoint after a call to
     * Triangulation::load().
     *
     * @note After mesh refinement all internal data structures stored in
     * the OfflineData object must be reinitialized with a call to
     * prepare() and the @p new_state_vector must be appropriately
     * initialized with the new Partitioner.
     */
    void project(StateVector<Number> &new_state_vector) override;

  private:
    //@}
    /**
     * @name Run time options
     */
    //@{

    Limiter limiter_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPIEnsemble &mpi_ensemble_;

    dealii::ObserverPointer<Interface::Discretization<dim>> discretization_;
    dealii::ObserverPointer<const Interface::OfflineData<dim, Number>>
        offline_data_;
    dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;
    dealii::ObserverPointer<const Interface::ParabolicSystem> parabolic_system_;

    //@}
    /**
     * @name Internal methods
     */
    //@{

    /**
     * Helper function reading in a state from the hyperbolic state vector.
     * The function translates the global index @p global_i into the local
     * range (operated on by this MPI rank) and automatically distributes
     * values to constrained degrees of freedom.
     *
     * @note Ordinarily distribute()ing constrained degrees of freedom
     * would happen with AffineConstraints<>::distribute() - but this
     * function is incompatible with our MultiComponentVector.
     */
    state_type read_tensor(const MultiComponentVector<Number> &U,
                           const dealii::types::global_dof_index global_i);

    /**
     * Helper function adding a supplied state to the hyperbolic state
     * vector. The function translates the global index @p global_i into
     * the local range (operated on by this MPI rank).
     */
    void add_tensor(MultiComponentVector<Number> &U,
                    const state_type &U_i,
                    const dealii::types::global_dof_index global_i);

    //@}
  };
} // namespace ryujin
