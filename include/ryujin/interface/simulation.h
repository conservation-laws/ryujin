//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/computing_timer.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/discretization/discretization.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/discretization/time_integrator.h>
#include <ryujin/interface/hyperbolic_module.h>
#include <ryujin/interface/hyperbolic_system.h>
#include <ryujin/interface/initial_values.h>
#include <ryujin/interface/parabolic_module.h>
#include <ryujin/interface/parabolic_system.h>
#include <ryujin/interface/postprocessor.h>
#include <ryujin/interface/solution_transfer.h>
#include <ryujin/interface/task.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <memory>

namespace ryujin
{
  namespace Interface
  {
    /**
     * The description-free interface of a Simulation: it stores the
     * core objects of a simulation, the discretization, the modules, the
     * time integrator, and the state vector, and has an accessor for
     * each. Tasks work with the Simulation through this interface.
     *
     * The Simulation is the primary task "time integration" of the
     * Scheduler that performs the time step in the step phase.
     *
     * The objects are constructed by a derived class, see the typed
     * Simulation, which is the only class that sees the hyperbolic and
     * the parabolic description.
     *
     * @ingroup Interfaces
     */
    template <int dim, typename Number = double>
    class Simulation : public Task<Number>
    {
    public:
      /**
       * Destructor.
       */
      virtual ~Simulation() = default;

      /**
       * @name Accessors
       */
      //@{

      ACCESSOR_READ_ONLY(mpi_ensemble)

      ACCESSOR(discretization)

      ACCESSOR_READ_ONLY(discretization)

      ACCESSOR_READ_ONLY(offline_data)

      ACCESSOR_READ_ONLY(hyperbolic_system)

      ACCESSOR_READ_ONLY(parabolic_system)

      ACCESSOR_READ_ONLY(initial_values)

      ACCESSOR_READ_ONLY(hyperbolic_module)

      ACCESSOR_READ_ONLY(parabolic_module)

      ACCESSOR_READ_ONLY(time_integrator)

      ACCESSOR_READ_ONLY(postprocessor)

      ACCESSOR(solution_transfer)

      ACCESSOR(state_vector)

      ACCESSOR_READ_ONLY(state_vector)

      //@}
      /**
       * @name Functions
       */
      //@{

      /**
       * Interpolate the analytic solution at time @p t into the state
       * vector @p analytic and populate precomputed values. In contrast to
       * TimeIntegrator::prepare_state_vector() no boundary conditions are
       * applied.
       */
      void interpolate_analytic_solution(StateVector<Number> &analytic,
                                         Number t) const
      {
        ComputingTimer::Scope scope(
            "time step [X]   - interpolate data vectors");

        hyperbolic_module_->reinit_state_vector(analytic);
        parabolic_module_->reinit_state_vector(analytic);
        analytic.state = initial_values_->interpolate_hyperbolic_vector(t);

        /* Populate precomputed values (without boundary conditions): */
        hyperbolic_module_->fill_precomputed_values(analytic);
      }

      //@}

    protected:
      /**
       * Constructor.
       */
      Simulation()
          : Task<Number>("time integration", TaskPhase::step)
      {
      }

      std::unique_ptr<MPIEnsemble> mpi_ensemble_;
      std::unique_ptr<Discretization<dim>> discretization_;
      std::unique_ptr<OfflineData<dim, Number>> offline_data_;

      /**
       * The systems and the initial values exist once per MPI ensemble.
       * A derived class owns them in MPIEnsembleContainer objects; the
       * pointers refer to the instance of the current ensemble.
       */
      const HyperbolicSystem *hyperbolic_system_ = nullptr;
      const ParabolicSystem *parabolic_system_ = nullptr;
      const InitialValues<dim, Number> *initial_values_ = nullptr;

      std::unique_ptr<HyperbolicModule<dim, Number>> hyperbolic_module_;
      std::unique_ptr<ParabolicModule<dim, Number>> parabolic_module_;
      std::unique_ptr<TimeIntegrator<dim, Number>> time_integrator_;
      std::unique_ptr<Postprocessor<dim, Number>> postprocessor_;
      std::unique_ptr<SolutionTransfer<dim, Number>> solution_transfer_;

      StateVector<Number> state_vector_;
    };
  } // namespace Interface
} // namespace ryujin
