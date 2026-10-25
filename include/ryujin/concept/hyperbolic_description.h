//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/concept/hyperbolic_system.h>
#include <ryujin/concept/indicator.h>
#include <ryujin/concept/limiter.h>
#include <ryujin/concept/wave_speed_estimator.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/interface/hyperbolic_module.h>
#include <ryujin/interface/initial_values.h>
#include <ryujin/interface/parabolic_system.h>

#include <deal.II/base/vectorization.h>

#include <concepts>
#include <string>

namespace ryujin
{
  namespace Concept
  {
    /**
     * The members of the hyperbolic description HD for the scalar or
     * vectorized number type Number.
     */
    template <typename HD>
    using system_type = typename HD::HyperbolicSystem;

    template <typename HD, int dim, typename Number>
    using initial_values_type =
        typename HD::template InitialValues<dim, Number>;

    template <typename HD, typename Number>
    using indicator_type = typename HD::template Indicator<scalar_type<Number>>;

    template <typename HD, typename Number>
    using limiter_type = typename HD::template Limiter<scalar_type<Number>>;

    template <typename HD, typename Number>
    using estimator_type =
        typename HD::template WaveSpeedEstimator<scalar_type<Number>>;

    template <typename Number>
    using vectorized_type = dealii::VectorizedArray<Number>;

    template <typename HD, int dim, typename Number>
    using hyperbolic_module_type =
        typename HD::template HyperbolicModule<dim, Number>;


    /**
     * The compute kernels of a hyperbolic description HD with hyperbolic
     * system System for one number type, which is scalar or vectorized.
     *
     * @ingroup Concepts
     */
    template <typename HD, typename System, int dim, typename Number>
    concept HyperbolicKernels =
        HyperbolicSystem<System, dim, Number> &&
        Indicator<indicator_type<HD, Number>, System, dim, Number> &&
        Limiter<limiter_type<HD, Number>, System, dim, Number> &&
        WaveSpeedEstimator<estimator_type<HD, Number>, System, dim, Number>;


    /**
     * A hyperbolic description used by the typed HyperbolicModule: a
     * hyperbolic system with an indicator, a limiter, and a wave speed
     * estimator, for scalar and vectorized numbers.
     *
     * @ingroup Concepts
     */
    template <typename HD, int dim, typename Number>
    concept HyperbolicModuleDescription =
        HyperbolicKernels<HD, system_type<HD>, dim, Number> &&
        HyperbolicKernels<HD, system_type<HD>, dim, vectorized_type<Number>>;


    /**
     * A hyperbolic description used by the Simulation: a hyperbolic
     * system for scalar and vectorized numbers, the initial values, a
     * limiter for the SolutionTransfer, and a hyperbolic module with the
     * constructor signature that the Simulation uses.
     *
     * The compute kernels of the hyperbolic module are not part of this
     * concept. A description whose hyperbolic module is the typed
     * HyperbolicModule additionally satisfies HyperbolicModuleDescription.
     *
     * @ingroup Concepts
     */
    template <typename HD, int dim, typename Number>
    concept HyperbolicDescription =
        HyperbolicSystem<system_type<HD>, dim, Number> &&
        HyperbolicSystem<system_type<HD>, dim, vectorized_type<Number>> &&
        std::derived_from<initial_values_type<HD, dim, Number>,
                          Interface::InitialValues<dim, Number>> &&
        std::constructible_from<initial_values_type<HD, dim, Number>,
                                const MPIEnsemble &,
                                const OfflineData<dim, Number> &,
                                const system_type<HD> &,
                                const Interface::ParabolicSystem &,
                                const std::string &> &&
        Limiter<limiter_type<HD, Number>, system_type<HD>, dim, Number> &&
        std::derived_from<hyperbolic_module_type<HD, dim, Number>,
                          Interface::HyperbolicModule<dim, Number>> &&
        std::constructible_from<hyperbolic_module_type<HD, dim, Number>,
                                const MPIEnsemble &,
                                const OfflineData<dim, Number> &,
                                const system_type<HD> &,
                                const initial_values_type<HD, dim, Number> &,
                                const std::string &>;
  } // namespace Concept
} // namespace ryujin
