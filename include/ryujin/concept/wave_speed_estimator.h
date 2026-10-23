//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/concept/hyperbolic_system.h>

#include <deal.II/base/tensor.h>

#include <concepts>
#include <string>

namespace ryujin
{
  namespace Concept
  {
    /**
     * The view of a wave speed estimator.
     *
     * @ingroup Concepts
     */
    template <typename V, int dim, typename Number>
    concept WaveSpeedEstimatorView =
        requires(const V &const_view,
                 const typename V::PrecomputedVectorView &precomputed,
                 const typename V::state_type &U,
                 const typename V::flux_contribution_type &flux,
                 const dealii::Tensor<1, dim, Number> &n_ij,
                 const unsigned int i,
                 const unsigned int *js) {
          {
            const_view.compute(precomputed, i, js, U, U, flux, flux, n_ij)
          } -> std::same_as<Number>;
        };


    /**
     * A wave speed estimator: constructed from the hyperbolic system and
     * the subsection of its run time parameters, it hands out views.
     *
     * @ingroup Concepts
     */
    template <typename T, typename System, int dim, typename Number>
    concept WaveSpeedEstimator =
        std::constructible_from<T, const System &, const std::string &> &&
        WaveSpeedEstimatorView<view_type<T, dim, Number>, dim, Number>;
  } // namespace Concept
} // namespace ryujin
