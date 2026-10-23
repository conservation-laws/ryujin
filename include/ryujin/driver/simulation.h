//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/computing_timer.h>
#include <ryujin/base/mpi_ensemble_container.h>
#include <ryujin/concept/hyperbolic_description.h>
#include <ryujin/concept/parabolic_description.h>
#include <ryujin/discretization/discretization.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/discretization/time_integrator.h>
#include <ryujin/interface/simulation.h>
#include <ryujin/interface/task.h>
#include <ryujin/typed/initial_values.h>
#include <ryujin/typed/postprocessor.h>
#include <ryujin/typed/solution_transfer.h>

#include <memory>

namespace ryujin
{
  /**
   * The Simulation constructs the core objects of a simulation for a given
   * pairing of a hyperbolic and a parabolic description.
   *
   * In addition to the objects stored in Interface::Simulation it owns
   * the per-ensemble containers of the hyperbolic system, the parabolic
   * system, and the initial values.
   *
   * The Simulation implements the primary task "time integration" of the
   * Scheduler: prepare() sets up the discretization and the state vector,
   * and execute() performs the time step.
   *
   * @ingroup Driver
   */
  template <typename HD, typename PD, int dim, typename Number = double>
    requires Concept::HyperbolicDescription<HD, dim, Number> &&
             Concept::ParabolicDescription<PD, HD, dim, Number>
  class Simulation final : public Interface::Simulation<dim, Number>
  {
  public:
    /**
     * @name Typedefs
     */
    //@{

    using HyperbolicSystem = typename HD::HyperbolicSystem;
    using ParabolicSystem = typename PD::ParabolicSystem;
    using HyperbolicModule =
        typename HD::template HyperbolicModule<dim, Number>;
    using ParabolicModule =
        typename PD::template ParabolicModule<HD, dim, Number>;

    //@}
    /**
     * @name Constructor and destructor
     */
    //@{

    /**
     * Constructor.
     */
    Simulation(const MPI_Comm &mpi_comm)
    {
      DeviceTimer::reinit();

      int n_ensembles = 1;
      if constexpr (has_n_mpi_ensembles_v<HD>)
        n_ensembles = HD::n_mpi_ensembles();

      this->mpi_ensemble_ =
          std::make_unique<MPIEnsemble>(mpi_comm, n_ensembles);
      const auto &mpi_ensemble = *this->mpi_ensemble_;

      hyperbolic_system_container_ =
          std::make_unique<MPIEnsembleContainer<HyperbolicSystem>>(
              mpi_ensemble, "/B - Equation");
      const HyperbolicSystem &hyperbolic_system =
          hyperbolic_system_container_->get();
      this->hyperbolic_system_ = &hyperbolic_system;

      parabolic_system_container_ =
          std::make_unique<MPIEnsembleContainer<ParabolicSystem>>(
              mpi_ensemble, "/B - Equation");
      const ParabolicSystem &parabolic_system =
          parabolic_system_container_->get();
      this->parabolic_system_ = &parabolic_system;

      this->discretization_ = std::make_unique<Discretization<dim>>(
          mpi_ensemble, "/C - Discretization");

      this->offline_data_ = std::make_unique<OfflineData<dim, Number>>(
          mpi_ensemble, *this->discretization_, "/D - OfflineData");
      const auto &offline_data = *this->offline_data_;

      initial_values_container_ = std::make_unique<
          MPIEnsembleContainer<InitialValues<HD, dim, Number>>>(
          mpi_ensemble,
          "/E - InitialValues",
          mpi_ensemble,
          offline_data,
          hyperbolic_system,
          parabolic_system);
      const InitialValues<HD, dim, Number> &initial_values =
          initial_values_container_->get();
      this->initial_values_ = &initial_values;

      auto hyperbolic_module =
          std::make_unique<HyperbolicModule>(mpi_ensemble,
                                             offline_data,
                                             hyperbolic_system,
                                             initial_values,
                                             "/F - HyperbolicModule");
      const auto &initial_precomputed =
          hyperbolic_module->initial_precomputed();
      this->hyperbolic_module_ = std::move(hyperbolic_module);

      this->parabolic_module_ =
          std::make_unique<ParabolicModule>(mpi_ensemble,
                                            offline_data,
                                            hyperbolic_system,
                                            parabolic_system,
                                            initial_values,
                                            "/G - ParabolicModule");

      this->time_integrator_ = std::make_unique<TimeIntegrator<dim, Number>>(
          mpi_ensemble,
          hyperbolic_system,
          parabolic_system,
          *this->hyperbolic_module_,
          *this->parabolic_module_,
          "/H - TimeIntegrator");

      this->postprocessor_ =
          std::make_unique<Postprocessor<HD, dim, Number>>(offline_data,
                                                           hyperbolic_system,
                                                           parabolic_system,
                                                           initial_precomputed);

      this->solution_transfer_ =
          std::make_unique<SolutionTransfer<HD, dim, Number>>(
              mpi_ensemble,
              *this->discretization_,
              offline_data,
              hyperbolic_system,
              parabolic_system,
              "/F - HyperbolicModule");
    }

    /**
     * Destructor. Destroys all objects in the reverse order of
     * construction because the objects observe each other.
     */
    ~Simulation() override
    {
      this->state_vector_ = StateVector<Number>();
      this->solution_transfer_.reset();
      this->postprocessor_.reset();
      this->time_integrator_.reset();
      this->parabolic_module_.reset();
      this->hyperbolic_module_.reset();
      this->initial_values_ = nullptr;
      initial_values_container_.reset();
      this->offline_data_.reset();
      this->discretization_.reset();
      this->parabolic_system_ = nullptr;
      parabolic_system_container_.reset();
      this->hyperbolic_system_ = nullptr;
      hyperbolic_system_container_.reset();
      this->mpi_ensemble_.reset();
    }

    //@}
    /**
     * @name Task
     */
    //@{

    /**
     * Prepare the discretization and the state vector for the current
     * mesh.
     *
     * If no triangulation has been created (or loaded from a checkpoint) yet,
     * create the mesh. If a state vector is attached to the triangulation
     * (after a mesh adaptation or a resume), project it onto the new mesh;
     * otherwise interpolate the initial values.
     */
    void prepare(TaskContext<Number> &context) override
    {
      auto &mpi_ensemble = *this->mpi_ensemble_;
      auto &discretization = *this->discretization_;
      auto &state_vector = this->state_vector_;

      if (!discretization.has_triangulation()) {
        if (mpi_ensemble.n_ensembles() > 1)
          context.info("setting up MPI ensemble");
        context.info("creating mesh and interpolating initial values");
        discretization.prepare(mpi_ensemble.base_name(context.base_name));
      }

      context.info("preparing compute kernels");
      discretization.update_mapping();
      this->offline_data_->prepare(
          this->hyperbolic_module_->problem_dimension(),
          this->hyperbolic_module_->n_precomputed_values());
      this->hyperbolic_module_->prepare();
      this->parabolic_module_->prepare();
      this->time_integrator_->prepare();

      this->hyperbolic_module_->reinit_state_vector(state_vector);
      this->parabolic_module_->reinit_state_vector(state_vector);

      auto &solution_transfer = *this->solution_transfer_;

      if (solution_transfer.has_handle()) {
        /* project the state vector if solution_transfer contains a handle: */
        solution_transfer.project(state_vector);
        solution_transfer.reset_handle();

      } else {
        /* Otherwise interpolate initial values: */
        ComputingTimer::Scope scope(
            "time step [X]   - interpolate data vectors");
        state_vector.state =
            this->initial_values_->interpolate_hyperbolic_vector();
      }

      debug_poison_invalid_values(state_vector, *this->offline_data_);

      /* Prepare the state vector for time stepping. */
      this->time_integrator_->prepare_state_vector(state_vector, context.t);
    }

    /**
     * Perform a time step from t to t + tau with the TimeIntegrator and
     * prepare the state vector at time t + tau.
     */
    void execute(TaskContext<Number> &context) override
    {
      auto &state_vector = this->state_vector_;

      context.tau = this->time_integrator_->step(
          state_vector, context.t, Number(0.), context.t_max - context.t);

      this->time_integrator_->prepare_state_vector(state_vector,
                                                   context.t + context.tau);
    }

    //@}

  private:
    /**
     * @name Internal data
     */
    //@{

    std::unique_ptr<MPIEnsembleContainer<HyperbolicSystem>>
        hyperbolic_system_container_;
    std::unique_ptr<MPIEnsembleContainer<ParabolicSystem>>
        parabolic_system_container_;
    std::unique_ptr<MPIEnsembleContainer<InitialValues<HD, dim, Number>>>
        initial_values_container_;

    //@}
  };
} // namespace ryujin
