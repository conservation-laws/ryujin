//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/interface/simulation.h>
#include <ryujin/typed/hyperbolic_module.h>
#include <ryujin/typed/initial_values.h>
#include <ryujin/typed/postprocessor.h>
#include <ryujin/typed/solution_transfer.h>

#include <ryujin/base/mpi_ensemble_container.h>
#include <ryujin/discretization/discretization.h>
#include <ryujin/discretization/mesh_adaptor.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/discretization/time_integrator.h>

#include <memory>

namespace ryujin
{
  /**
   * The Simulation constructs all objects of a computation for the
   * pairing of a hyperbolic and a parabolic description. It is the only
   * class that sees both descriptions.
   *
   * In addition to the objects stored in Interface::Simulation it owns
   * the per-ensemble containers of the hyperbolic system, the parabolic
   * system, and the initial values, and the postprocessor that the
   * description-free modules share.
   *
   * @ingroup TimeLoop
   */
  template <typename HyperbolicDescription,
            typename ParabolicDescription,
            int dim,
            typename Number = double>
  class Simulation final : public Interface::Simulation<dim, Number>
  {
  public:
    /**
     * @name Typedefs
     */
    //@{

    using HD = HyperbolicDescription;
    using PD = ParabolicDescription;

    using HyperbolicSystem = typename HD::HyperbolicSystem;
    using ParabolicSystem = typename PD::ParabolicSystem;
    using ParabolicModule =
        typename PD::template ParabolicModule<HD, dim, Number>;

    //@}
    /**
     * @name Constructor and destructor
     */
    //@{

    /**
     * Constructor. Constructs all objects in the order of the
     * subsections "/B - Equation" to "/L - ErrorEvaluation".
     */
    Simulation(const MPI_Comm &mpi_comm)
    {
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
          std::make_unique<HyperbolicModule<HD, dim, Number>>(
              mpi_ensemble,
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

      postprocessor_ =
          std::make_unique<Postprocessor<HD, dim, Number>>(offline_data,
                                                           hyperbolic_system,
                                                           parabolic_system,
                                                           initial_precomputed);

      this->mesh_adaptor_ = std::make_unique<MeshAdaptor<dim, Number>>(
          mpi_ensemble, offline_data, *postprocessor_, "/I - MeshAdaptor");

      this->solution_transfer_ =
          std::make_unique<SolutionTransfer<HD, dim, Number>>(
              mpi_ensemble,
              *this->discretization_,
              offline_data,
              hyperbolic_system,
              parabolic_system,
              "/I - MeshAdaptor");

      this->vtu_output_ =
          std::make_unique<VTUOutput<dim, Number>>(mpi_ensemble,
                                                   offline_data,
                                                   hyperbolic_system,
                                                   *postprocessor_,
                                                   "/J - VTUOutput");

      this->quantities_ =
          std::make_unique<Quantities<dim, Number>>(mpi_ensemble,
                                                    offline_data,
                                                    hyperbolic_system,
                                                    *postprocessor_,
                                                    "/K - Quantities");

      this->error_evaluation_ = std::make_unique<ErrorEvaluation<dim, Number>>(
          mpi_ensemble,
          offline_data,
          hyperbolic_system,
          *postprocessor_,
          "/L - ErrorEvaluation");
    }

    /**
     * Destructor. Destroys all objects in the reverse order of
     * construction because the objects observe each other.
     */
    ~Simulation() override
    {
      this->error_evaluation_.reset();
      this->quantities_.reset();
      this->vtu_output_.reset();
      this->solution_transfer_.reset();
      this->mesh_adaptor_.reset();
      postprocessor_.reset();
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
    std::unique_ptr<Postprocessor<HD, dim, Number>> postprocessor_;

    //@}
  };
} // namespace ryujin
