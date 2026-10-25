//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/interface/hyperbolic_module.h>
#include <ryujin/interface/hyperbolic_system.h>
#include <ryujin/interface/parabolic_module.h>
#include <ryujin/interface/parabolic_system.h>
#include <ryujin/interface/time_integrator.h>

#include <concepts>
#include <string>

namespace ryujin
{
  namespace Concept
  {
    /**
     * A time integrator TI: it implements Interface::TimeIntegrator and
     * has the constructor signature that the Simulation uses.
     *
     * @ingroup Concepts
     */
    template <typename TI, int dim, typename Number>
    concept TimeIntegrator =
        std::derived_from<TI, Interface::TimeIntegrator<dim, Number>> &&
        std::constructible_from<
            TI,
            const MPIEnsemble &,
            const Interface::HyperbolicSystem &,
            const Interface::ParabolicSystem &,
            const Interface::HyperbolicModule<dim, Number> &,
            const Interface::ParabolicModule<dim, Number> &,
            const std::string &>;
  } // namespace Concept
} // namespace ryujin
