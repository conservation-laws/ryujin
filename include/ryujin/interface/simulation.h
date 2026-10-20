//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/discretization/discretization.h>
#include <ryujin/discretization/mesh_adaptor.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/discretization/time_integrator.h>
#include <ryujin/interface/hyperbolic_module.h>
#include <ryujin/interface/hyperbolic_system.h>
#include <ryujin/interface/initial_values.h>
#include <ryujin/interface/parabolic_module.h>
#include <ryujin/interface/parabolic_system.h>
#include <ryujin/interface/solution_transfer.h>
#include <ryujin/output/error_evaluation.h>
#include <ryujin/output/quantities.h>
#include <ryujin/output/vtu_output.h>

#include <memory>

namespace ryujin
{
  namespace Interface
  {
    /**
     * The description-free interface of a Simulation: it stores all
     * objects that the Scheduler works with and has an accessor for each.
     *
     * The objects are constructed by a derived class, see the typed
     * Simulation, which is the only class that sees the hyperbolic and
     * the parabolic description.
     *
     * @ingroup Interfaces
     */
    template <int dim, typename Number = double>
    class Simulation
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

      ACCESSOR(offline_data)

      ACCESSOR_READ_ONLY(hyperbolic_system)

      ACCESSOR_READ_ONLY(parabolic_system)

      ACCESSOR_READ_ONLY(initial_values)

      ACCESSOR(hyperbolic_module)

      ACCESSOR(parabolic_module)

      ACCESSOR(time_integrator)

      ACCESSOR(mesh_adaptor)

      ACCESSOR(solution_transfer)

      ACCESSOR(vtu_output)

      ACCESSOR(quantities)

      ACCESSOR(error_evaluation)

      //@}

    protected:
      /**
       * Constructor.
       */
      Simulation() = default;

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
      std::unique_ptr<MeshAdaptor<dim, Number>> mesh_adaptor_;
      std::unique_ptr<SolutionTransfer<dim, Number>> solution_transfer_;
      std::unique_ptr<VTUOutput<dim, Number>> vtu_output_;
      std::unique_ptr<Quantities<dim, Number>> quantities_;
      std::unique_ptr<ErrorEvaluation<dim, Number>> error_evaluation_;
    };
  } // namespace Interface
} // namespace ryujin
