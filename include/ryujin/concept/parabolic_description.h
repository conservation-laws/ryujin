//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/concept/parabolic_system.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/interface/parabolic_module.h>
#include <ryujin/typed/initial_values.h>

#include <concepts>
#include <string>

namespace ryujin
{
  namespace Concept
  {
    /**
     * The parabolic module of the parabolic description PD paired with
     * the hyperbolic description HD.
     */
    template <typename PD, typename HD, int dim, typename Number>
    using parabolic_module_type =
        typename PD::template ParabolicModule<HD, dim, Number>;


    /**
     * A parabolic description that can be paired with the hyperbolic
     * description HD: a parabolic system, and a parabolic module with the
     * constructor signature that the Simulation uses.
     *
     * @ingroup Concepts
     */
    template <typename T, typename HD, int dim, typename Number>
    concept ParabolicDescription =
        ParabolicSystem<typename T::ParabolicSystem> &&
        std::derived_from<parabolic_module_type<T, HD, dim, Number>,
                          Interface::ParabolicModule<dim, Number>> &&
        std::constructible_from<
            parabolic_module_type<T, HD, dim, Number>,
            const MPIEnsemble &,
            const OfflineData<dim, Number> &,
            const typename HD::HyperbolicSystem &,
            const typename T::ParabolicSystem &,
            const InitialValues<typename HD::HyperbolicSystem, dim, Number> &,
            const std::string &>;
  } // namespace Concept
} // namespace ryujin
