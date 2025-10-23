//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception or LGPL-2.1-or-later
// Copyright (C) 2020 - 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "mpi_ensemble.h"
#include "observer_pointer.h"
#include "offline_data.h"
#include "state_vector.h"

#include <deal.II/base/parameter_acceptor.h>

namespace ryujin
{
  /**
   * The SolutionTransfer class is an adaptation of the
   * parallel::distributed::SolutionTransfer class and method implemented
   * in deal.II. This class is, in contrast to the deal.II version,
   * specifically taylored to our needs:
   *  - it uses the MultiComponentVector directly,
   *  - it implements a local mass matrix projection with convex limiting.
   *
   * @ingroup Mesh
   */
  template <typename Description, int dim, typename Number = double>
  class SolutionTransfer : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @name Typedefs and constexpr constants
     */
    //@{

    using HyperbolicSystem = typename Description::HyperbolicSystem;
    using ParabolicSystem = typename Description::ParabolicSystem;

    using View =
        typename Description::template HyperbolicSystemView<dim, Number>;

    static constexpr auto problem_dimension = View::problem_dimension;

    using state_type = typename View::state_type;

    using StateVector = typename View::StateVector;
    using HyperbolicVector = typename View::HyperbolicVector;

    using ScalarVector = Vectors::ScalarVector<Number>;

    /**
     * The number of stored entries in the bounds array.
     */
    static constexpr unsigned int n_bounds =
        Description::template Limiter<dim, Number>::n_bounds;

    /**
     * Array type used to store accumulated bounds.
     */
    using Bounds = typename Description::template Limiter<dim, Number>::Bounds;

    //@}
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor
     */
    SolutionTransfer(const MPIEnsemble &mpi_ensemble,
                     const OfflineData<dim, Number> &offline_data,
                     const HyperbolicSystem &hyperbolic_system,
                     const ParabolicSystem &parabolic_system,
                     const std::string &subsection = "/SolutionTransfer");

    /**
     * Destructor
     */
    ~SolutionTransfer() override = default;

    //@}
    /**
     * @name Methods for solution transfer after mesh adaptation and
     * serialization/deserialization.
     */
    //@{

    /**
     * Prepare projection (and serialization) by registering a call back to
     * all cells of the distributed triangulation. The call back attaches
     * all necessary data to the triangulation to perform a projection
     * after mesh adaptation, or a Triangulation::store()/load() operation.
     */
    void prepare_projection(const StateVector &old_state_vector);

    /**
     * Return the handle associated with the call back that was set by
     * prepare_projection() and the associated data attached to the
     * triangulation.
     */
    unsigned int get_handle() const
    {
      Assert(
          handle_ != dealii::numbers::invalid_unsigned_int,
          dealii::ExcMessage("Invalid handle: Cannot retrieve a valid "
                             "handle. get_handle() can only be called after a "
                             "call to prepare_projection(), or set_handle()."));
      return handle_;
    }

    /**
     * Set the handle associated data attached to the triangulation. This
     * function has to be called after a Triangulation::load() operation
     * and prior to the SolutionTransfer::project().
     */
    void set_handle(unsigned int handle)
    {
      Assert(
          handle_ == dealii::numbers::invalid_unsigned_int,
          dealii::ExcMessage("Invalid state: Cannot set handle because we "
                             "already have a valid handle due to a prior call "
                             "to prepare_projection(), or set_handle()."));
      handle_ = handle;
    }

    /**
     * Return the handle associated with the call back that was set by
     * prepare_projection() and the associated data attached to the
     * triangulation.
     */
    void reset_handle()
    {
      handle_ = dealii::numbers::invalid_unsigned_int;
    }

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
    void project(StateVector &new_state_vector);

  private:
    //@}
    /**
     * @name Run time options
     */
    //@{

    typename Description::template Limiter<dim, Number>::Parameters
        limiter_parameters_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPIEnsemble &mpi_ensemble_;

    dealii::ObserverPointer<const OfflineData<dim, Number>> offline_data_;
    dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;
    dealii::ObserverPointer<const ParabolicSystem> parabolic_system_;

    unsigned int handle_;

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
    state_type get_tensor(const HyperbolicVector &U,
                          const dealii::types::global_dof_index global_i);

    /**
     * Helper function adding a supplied state to the hyperbolic state
     * vector. The function translates the global index @p global_i into
     * the local range (operated on by this MPI rank).
     */
    void add_tensor(HyperbolicVector &U,
                    const state_type &U_i,
                    const dealii::types::global_dof_index global_i);
  };
} // namespace ryujin
