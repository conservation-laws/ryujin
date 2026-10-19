//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/interface/offline_data.h>
#include <ryujin/interface/parabolic_module.h>
#include <ryujin/stub/parabolic_system.h>
#include <ryujin/typed/initial_values.h>

#include <deal.II/base/parameter_acceptor.h>

namespace ryujin
{
  /**
   * A stub parabolic solver that models the identity.
   *
   * @ingroup Typed
   */
  template <typename HyperbolicDescription, int dim, typename Number>
  class StubParabolicModule final
      : public Interface::ParabolicModule<dim, Number>
  {
  public:
    /**
     * @name Typedefs and constexpr constants
     */
    //@{

    using StageVectors =
        typename Interface::ParabolicModule<dim, Number>::StageVectors;

    using StageWeights =
        typename Interface::ParabolicModule<dim, Number>::StageWeights;

    using HyperbolicSystem = typename HyperbolicDescription::HyperbolicSystem;
    using ParabolicSystem = StubParabolicSystem;

    using View = typename HyperbolicSystem::template View<dim, Number>;

    //@}
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor.
     */
    StubParabolicModule(
        const MPIEnsemble & /*mpi_ensemble*/,
        const Interface::OfflineData<dim, Number> & /*offline_data*/,
        const HyperbolicSystem & /*hyperbolic_system*/,
        const ParabolicSystem & /*parabolic_system*/,
        const InitialValues<HyperbolicDescription, dim, Number>
            & /*initial_values*/,
        const std::string &subsection = "StubParabolicModule")
        : Interface::ParabolicModule<dim, Number>(subsection)
    {
    }

    /**
     * Prepare time stepping. A call to @p prepare() allocates temporary
     * storage and is necessary before any of the following time-stepping
     * functions can be called.
     */
    void prepare() override
    {
      // do nothing
    }

    //@}
    /**
     * @name Functions for performing explicit time steps
     */
    //@{

    /**
     * (Re)initialize the parabolic state vector component of the state
     * vector.
     *
     * @note This routine does not modify the hyperbolic state vector or
     * the precomputed vector component.
     */
    void
    reinit_state_vector(StateVector<Number> & /*state_vector*/) const override
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
    void prepare_state_vector(StateVector<Number> & /*state_vector*/,
                              Number /*t*/) const override
    {
      Assert(false,
             dealii::ExcMessage("The parabolic system is the identity. This "
                                "function should have never been called."));
    }

    /**
     * Given a reference to a previous state vector @p old_state_vector at
     * time @p old_t and a time-step size @p tau perform an implicit
     * backward euler step (and store the result in @p new_state_vector).
     */
    void backward_euler_step(const StateVector<Number> &old_state_vector,
                             const Number /*old_t*/,
                             StageVectors /*stage_state_vectors*/,
                             StageWeights /*stage_weights*/,
                             StateVector<Number> &new_state_vector,
                             Number /*tau*/) const override
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
    void crank_nicolson_step(const StateVector<Number> &old_state_vector,
                             const Number /*old_t*/,
                             StateVector<Number> &new_state_vector,
                             Number /*tau*/) const override
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
    void print_solver_statistics(std::ostream & /*output*/) const override
    {
      // do nothing
    }

    //@}
  };


  /**
   * The parabolic description of a purely hyperbolic equation: the
   * identity as parabolic system and a stub parabolic module. It can be
   * paired with every hyperbolic description.
   *
   * @ingroup Typed
   */
  struct StubParabolicDescription {
    using ParabolicSystem = StubParabolicSystem;

    template <typename HyperbolicDescription, int dim, typename Number = double>
    using ParabolicModule =
        StubParabolicModule<HyperbolicDescription, dim, Number>;
  };
} // namespace ryujin
